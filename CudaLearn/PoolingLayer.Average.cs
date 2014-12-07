using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class AveragePoolingLayerConfiguration : PoolingLayerConfiguration
    {
        public AveragePoolingLayerConfiguration(int kernelSize, int stride = 1, int padding = 0)
            : this(new Size(kernelSize, kernelSize), new Size(stride, stride), new Size(padding, padding))
        { }

        public AveragePoolingLayerConfiguration( Size kernel, Size stride, Size padding = default(Size)) : base ( LayerType.AveragePooling)
        {
            this.Kernel = kernel;
            this.Stride = stride;
            this.Padding = padding;
        }

        public AveragePoolingLayerConfiguration()
            : base(LayerType.AveragePooling)
        { }
    }

    public class AveragePoolingLayer : PoolingLayer<AveragePoolingLayerConfiguration>
    {
        public AveragePoolingLayer(int kernelSize, int stride = 1, int padding = 0)
            : this(new AveragePoolingLayerConfiguration(kernelSize, stride, padding))
        { }

        public AveragePoolingLayer(Size kernel, Size stride, Size padding = default(Size))
            : this(new AveragePoolingLayerConfiguration(kernel, stride, padding))
        { }

        public AveragePoolingLayer()
            : this(new AveragePoolingLayerConfiguration())
        { }

        public AveragePoolingLayer(AveragePoolingLayerConfiguration config)
            : base(config)
        { }


        public override void Setup(IList<Blob> bottom, IList<Blob> top)
        {
            CheckSizeParameters ();

            this._maxTopBlobs = 1;

            base.Setup(bottom, top);

            int channels = bottom[0].Channels;
            top[0].Reshape(bottom[0].Num, channels, Pooled.Height, Pooled.Width);    
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


        protected override double ForwardCpu(IList<Blob> bottom, IList<Blob> top)
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

            // Zero the output
            topData.Map(v => 0, result: topData);

            // Main loop
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

                            for (int h = hstart; h < hend; h++)
                            {
                                for (int w = wstart; w < wend; w++)
                                    topData[topOffset + ph * Pooled.Width + pw] += bottomData[bottomOffset + h * width + w];
                            }
                            topData[topOffset + ph * Pooled.Width + pw] /= pool_size;
                        }
                    }

                    bottomOffset += bottom[0].Offset(0, 1);
                    topOffset += top[0].Offset(0, 1);
                }
            }

            return 0;
        }

        protected override void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            var bottomDiff = bottom[0].Diff;
            var topDiff = top[0].Diff;

            int num = bottom[0].Num;
            int channels = bottom[0].Channels;
            int height = bottom[0].Height;
            int width = bottom[0].Width;

            Size padding = this.Parameters.Padding;
            Size stride = this.Parameters.Stride;
            Size kernel = this.Parameters.Kernel;

            // Zero the output
            bottomDiff.Map(v => 0, result: bottomDiff);

            // Main loop            
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

                            int pos = topOffset + ph * Pooled.Width + pw;
                            for (int h = hstart; h < hend; h++)
                            {
                                for (int w = wstart; w < wend; w++)
                                    bottomDiff[bottomOffset + h * width + w] += topDiff[pos] / pool_size;
                            }
                        }
                    }

                    bottomOffset += bottom[0].Offset(0, 1);
                    topOffset += top[0].Offset(0, 1);
                }
            }
        }
    }
}
