using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class StocasticPoolingLayerConfiguration : PoolingLayerConfiguration
    {
        public StocasticPoolingLayerConfiguration(int kernelSize, int stride = 1, int padding = 0)
            : this(new Size(kernelSize, kernelSize), new Size(stride, stride), new Size(padding, padding))
        { }

        public StocasticPoolingLayerConfiguration(Size kernel, Size stride, Size padding = default(Size))
            : base(LayerType.StocasticPooling)
        {
            this.Kernel = kernel;
            this.Stride = stride;
            this.Padding = padding;
        }

        public StocasticPoolingLayerConfiguration()
            : base(LayerType.StocasticPooling)
        { }
    }

    public class StocasticPoolingLayer : PoolingLayer<StocasticPoolingLayerConfiguration>
    {
        private Vector<double> randomIndexes;

        public StocasticPoolingLayer(int kernelSize, int stride = 1, int padding = 0)
            : this(new StocasticPoolingLayerConfiguration(kernelSize, stride, padding))
        { }

        public StocasticPoolingLayer(Size kernel, Size stride, Size padding = default(Size))
            : this(new StocasticPoolingLayerConfiguration(kernel, stride, padding))
        { }

        public StocasticPoolingLayer()
            : this(new StocasticPoolingLayerConfiguration())
        { }

        public StocasticPoolingLayer(StocasticPoolingLayerConfiguration config)
            : base(config)
        { }


        public override void Setup(TensorCollection bottom, TensorCollection top)
        {
            CheckSizeParameters();

            this._maxTopBlobs = 1;

            base.Setup(bottom, top);

            int channels = bottom[0].Channels;
            top[0].Reshape(bottom[0].Num, channels, Pooled.Height, Pooled.Width);

            using (var topCpu = top[0].OnCpu())
            {
                this.randomIndexes = Vector<double>.Build.SameAs(topCpu.Data);
                var distribution = new ContinuousUniform(0, 1);
                randomIndexes.MapInplace(x => distribution.Sample(), Zeros.Include);   
            }                    
        }

        private void CheckSizeParameters()
        {
            Guard.That(() => this.Parameters.Kernel.Width).IsPositive();
            Guard.That(() => this.Parameters.Kernel.Height).IsPositive();
            Guard.That(() => this.Parameters.Kernel.Depth).IsEqual(0);

            Guard.That(() => this.Parameters.Stride.Width).IsPositive();
            Guard.That(() => this.Parameters.Stride.Height).IsPositive();
            Guard.That(() => this.Parameters.Stride.Depth).IsEqual(0);

            Guard.That(() => this.Parameters.Padding.Width).IsEqual(0);
            Guard.That(() => this.Parameters.Padding.Height).IsEqual(0);
            Guard.That(() => this.Parameters.Padding.Depth).IsEqual(0);
        }

        internal override double ForwardCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
        {
            if (this.Phase == PhaseType.Train)
                return ForwardTrainCpu(bottom, top);
            else
                return ForwardTestCpu(bottom, top);
        }

        protected double ForwardTrainCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            int num = bottom[0].Num;
            int channels = bottom[0].Channels;
            int height = bottom[0].Height;
            int width = bottom[0].Width;

            Size padding = this.Parameters.Padding;
            Size stride = this.Parameters.Stride;
            Size kernel = this.Parameters.Kernel;

            // Main loop         
            topData.MapIndexed((index, _) =>
                {
                    int pw = index % Pooled.Width;
                    int ph = (index / Pooled.Width) % Pooled.Height;
                    int c = (index / Pooled.Width / Pooled.Height) % channels;
                    int n = index / Pooled.Width / Pooled.Height / channels;

                    int hstart = ph * stride.Height;
                    int hend = Math.Min(hstart + kernel.Height, height);
                    int wstart = pw * stride.Width;
                    int wend = Math.Min(wstart + kernel.Width, width);

                    int bottomOffset = (n * channels + c) * height * width;

                    // First pass: get sum
                    double cumulativeSum = 0;
                    for (int h = hstart; h < hend; h++)
                        for (int w = wstart; w < wend; w++)
                            cumulativeSum += bottomData[bottomOffset + h * width + w];

                    double threshold = this.randomIndexes[index] * cumulativeSum;

                    // Second pass: get value, and set index.
                    for (int h = hstart; h < hend; ++h)
                    {
                        for (int w = wstart; w < wend; ++w)
                        {
                            cumulativeSum += bottomData[bottomOffset + h * width + w];
                            if (cumulativeSum >= threshold)
                            {
                                this.randomIndexes[index] = ((n * channels + c) * height + h) * width + w;
                                return bottomData[bottomOffset + h * width + w];
                            }
                        }
                    }

                    this.randomIndexes[index] = ((n * channels + c) * height + (hend - 1)) * width + (wend - 1);
                    return bottomData[bottomOffset + (hend - 1) * width + (wend - 1)];
                }, topData, Zeros.Include);

            return 0;
        }


        protected double ForwardTestCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            int channels = bottom[0].Channels;
            int height = bottom[0].Height;
            int width = bottom[0].Width;

            Size stride = this.Parameters.Stride;
            Size kernel = this.Parameters.Kernel;

            // Main loop         
            topData.MapIndexed((index, _) =>
            {
                int pw = index % Pooled.Width;
                int ph = (index / Pooled.Width) % Pooled.Height;
                int c = (index / Pooled.Width / Pooled.Height) % channels;
                int n = index / Pooled.Width / Pooled.Height / channels;

                int hstart = ph * stride.Height;
                int hend = Math.Min(hstart + kernel.Height, height);
                int wstart = pw * stride.Width;
                int wend = Math.Min(wstart + kernel.Width, width);

                int bottomOffset = (n * channels + c) * height * width;

                double cumulativeSum = double.Epsilon;
                double cumulativeValues = 0;
                for (int h = hstart; h < hend; h++)
                {
                    for (int w = wstart; w < wend; w++)
                    {
                        double value = bottomData[bottomOffset + h * width + w];
                        cumulativeSum += value;
                        cumulativeValues += value * value;
                    }
                }                 

                return cumulativeValues / cumulativeSum;
            }, topData, Zeros.Include);

            return 0;
        }

        internal override void BackwardCpu(CpuTensorScopeCollection top, IList<bool> propagateDown, CpuTensorScopeCollection bottom)
        {
            var bottomDiff = bottom[0].Diff;
            var topDiff = top[0].Diff;

            int channels = bottom[0].Channels;
            int height = bottom[0].Height;
            int width = bottom[0].Width;

            Size stride = this.Parameters.Stride;
            Size kernel = this.Parameters.Kernel;

            // Zero the output
            bottomDiff.Map(v => 0, result: bottomDiff);

            // Main loop            
            for (int index = 0; index < bottom.Count; index++)
            {
                // find out the local index
                // find out the local offset
                int w = index % width;
                int h = (index / width) % height;
                int c = (index / width / height) % channels;
                int n = index / width / height / channels;

                int phstart = (h < kernel.Height) ? 0 : (h - kernel.Height) / stride.Height + 1;
                int phend = Math.Min(h / stride.Height + 1, Pooled.Height);
                int pwstart = (w < kernel.Width) ? 0 : (w - kernel.Width) / stride.Width + 1;
                int pwend = Math.Min(w / stride.Width + 1, Pooled.Width);

                int topOffset = (n * channels + c) * Pooled.Height * Pooled.Width;

                double gradient = 0;

                for (int ph = phstart; ph < phend; ++ph)
                {
                    for (int pw = pwstart; pw < pwend; ++pw)
                    {
                        if (index == randomIndexes[topOffset + ph * Pooled.Width + pw])
                            gradient += topDiff[topOffset + ph * Pooled.Width + pw];
                    }
                }

                bottomDiff[index] = gradient;
            }
        }
    }

}
