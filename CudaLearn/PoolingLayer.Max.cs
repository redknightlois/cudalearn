using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class MaxPoolingLayerConfiguration : PoolingLayerConfiguration
    {
        public MaxPoolingLayerConfiguration(int kernelSize, int stride = 1, int padding = 0)
            : this(new Size(kernelSize, kernelSize), new Size(stride, stride), new Size(padding, padding))
        { }

        public MaxPoolingLayerConfiguration(Size kernel, Size stride, Size padding = default(Size))
            : base(LayerType.MaxPooling)
        {
            this.Kernel = kernel;
            this.Stride = stride;
            this.Padding = padding;
        }

        public MaxPoolingLayerConfiguration()
            : base(LayerType.MaxPooling)
        { }
    }

    public class MaxPoolingLayer : PoolingLayer<MaxPoolingLayerConfiguration>
    {
        private Vector<double> maxIndexes;

        public MaxPoolingLayer(int kernelSize, int stride = 1, int padding = 0)
            : this(new MaxPoolingLayerConfiguration(kernelSize, stride, padding))
        { }

        public MaxPoolingLayer(Size kernel, Size stride, Size padding = default(Size))
            : this(new MaxPoolingLayerConfiguration(kernel, stride, padding))
        { }

        public MaxPoolingLayer()
            : this(new MaxPoolingLayerConfiguration())
        { }

        public MaxPoolingLayer(MaxPoolingLayerConfiguration config)
            : base(config)
        { }

        public override void Setup(TensorCollection bottom, TensorCollection top)
        {
            CheckSizeParameters();

            this._maxTopBlobs = 2;

            base.Setup(bottom, top);

            int num = bottom[0].Num;
            int channels = bottom[0].Channels;

            foreach( var item in top )
                item.Reshape(num, channels, Pooled.Height, Pooled.Width);

            using (var topCpu = top[0].OnCpu())
            {
                this.maxIndexes = Vector<double>.Build.SameAs(topCpu.Data);
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

            Guard.That(() => this.Parameters.Padding.Width).IsGreaterOrEqualThan(0);
            Guard.That(() => this.Parameters.Padding.Height).IsGreaterOrEqualThan(0);
            Guard.That(() => this.Parameters.Padding.Depth).IsEqual(0);
        }

        internal override double ForwardCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
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

            // Initialize 
            Vector<double> outputMask;

            bool useTopMask = top.Count > 1;
            if ( useTopMask )
                outputMask = top[1].Data;
            else
                outputMask = this.maxIndexes;

            outputMask.Map(x => -1f, outputMask, Zeros.Include);
            topData.Map(x => double.MinValue, topData, Zeros.Include);

            // The main loop
            int bottomOffset = 0;
            int topOffset = 0;
            for (int n = 0; n < num; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int ph = 0; ph < Pooled.Height; ph++)
                    {
                        for (int pw = 0; pw < Pooled.Width; pw++)
                        {
                            int hstart = ph * stride.Height - padding.Height;
                            int wstart = pw * stride.Width - padding.Width;
                            int hend = Math.Min(hstart + kernel.Height, height + padding.Height);
                            int wend = Math.Min(wstart + kernel.Width, width + padding.Width);
                            int pool_size = (hend - hstart) * (wend - wstart);

                            hstart = Math.Max(hstart, 0);
                            wstart = Math.Max(wstart, 0);
                            hend = Math.Min(hend, height);
                            wend = Math.Min(wend, width);

                            int poolIndex = ph * Pooled.Width + pw;
                            for (int h = hstart; h < hend; h++)
                            {
                                for (int w = wstart; w < wend; w++)
                                {
                                    int index = h * width + w;
                                    if (bottomData[bottomOffset + index] > topData[topOffset + poolIndex])
                                    {
                                        topData[topOffset + poolIndex] = bottomData[bottomOffset + index];
                                        outputMask[topOffset + poolIndex] = index;
                                    }
                                }                                    
                            }
                        }
                    }

                    bottomOffset += bottom[0].Offset(0, 1);
                    topOffset += top[0].Offset(0, 1);
                }
            }

            return 0;  
        }

        internal override void BackwardCpu(CpuTensorScopeCollection top, IList<bool> propagateDown, CpuTensorScopeCollection bottom)
        {
            if (!propagateDown[0])
                return;

            var bottomData = bottom[0].Data;
            var bottomDiff = bottom[0].Diff;
            var topData = top[0].Data;
            var topDiff = top[0].Diff;

            int count = bottom[0].Count;

            int num = bottom[0].Num;
            int channels = bottom[0].Channels;
            int height = bottom[0].Height;
            int width = bottom[0].Width;            

            Size padding = this.Parameters.Padding;
            Size stride = this.Parameters.Stride;
            Size kernel = this.Parameters.Kernel;

            // Zero the output
            bottomDiff.Map(v => 0, result: bottomDiff);

            // Initialize 
            Vector<double> inputMask;
            bool useTopMask = top.Count > 1;
            if (useTopMask)
                inputMask = top[1].Data;
            else
                inputMask = this.maxIndexes;

            // Main loop            
            for (int index = 0; index < count; index++)
            {
                int w = index % width;
                int h = (index / width) % height;
                int c = (index / width / height) % channels;
                int n = index / width / height / channels;

                int phstart = (h + padding.Height < kernel.Height) ? 0 : (h + padding.Height - kernel.Height) / stride.Height + 1;
                int phend = Math.Min((h + padding.Height) / stride.Height + 1, Pooled.Height);
                int pwstart = (w + padding.Width < kernel.Width) ? 0 : (w + padding.Width - kernel.Width) / stride.Width + 1;
                int pwend = Math.Min((w + padding.Width) / stride.Width + 1, Pooled.Width);

                int topOffset = (n * channels + c) * Pooled.Height * Pooled.Width;

                double bottomDatum = bottomData[index];

                double gradient = 0;
                for (int ph = phstart; ph < phend; ++ph)
                {
                    for (int pw = pwstart; pw < pwend; ++pw)
                    {
                        int topIndex = ph * Pooled.Width + pw;

                        if (bottomDatum == topData[topOffset + topIndex])
                            gradient += topDiff[topOffset + topIndex];
                    }
                }

                bottomDiff[index] = gradient;
            }
        }
    }
}
