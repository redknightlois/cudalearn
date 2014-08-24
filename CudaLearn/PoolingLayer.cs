using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class PoolingLayerConfiguration : LayerConfiguration
    {
        protected PoolingLayerConfiguration(LayerType type)
            : base(type)
        {
            this.Kernel = new Size(2, 2);
            this.Stride = new Size(1, 1);
            this.Padding = new Size();
        }

        public Size Kernel { get; set; }

        public Size Stride { get; set; }

        public Size Padding { get; set; }
    }

    public abstract class PoolingLayer<TConfiguration> : Layer<TConfiguration> where TConfiguration : PoolingLayerConfiguration
    {
        protected PoolingLayer(TConfiguration param)
            : base(param)
        {
        }

        public override void Setup(IList<Blob> bottom, IList<Blob> top)
        {
            base.Setup(bottom, top);

            int height = bottom[0].Height;
            int width = bottom[0].Width;

            int pooledHeight = (int)(Math.Ceiling((float)(height + 2 * this.Parameters.Padding.Height - this.Parameters.Kernel.Height) / this.Parameters.Stride.Height) + 1);
            int pooledWidth = (int)(Math.Ceiling((float)(width + 2 * this.Parameters.Padding.Width - this.Parameters.Kernel.Width) / this.Parameters.Stride.Width) + 1);

            Debug.Assert((pooledHeight - 1) * this.Parameters.Stride.Height <= height + this.Parameters.Padding.Height);
            Debug.Assert((pooledWidth - 1) * this.Parameters.Stride.Width <= width + this.Parameters.Padding.Width);

            if (this.Parameters.Padding.Height != 0 || this.Parameters.Padding.Width != 0)
            {
                // If we have padding, ensure that the last pooling starts strictly
                // inside the image (instead of at the padding); otherwise clip the last.
                if ((pooledHeight - 1) * this.Parameters.Stride.Height >= height + this.Parameters.Padding.Height)
                    pooledHeight--;

                if ((pooledWidth - 1) * this.Parameters.Stride.Width >= width + this.Parameters.Padding.Width)
                    pooledWidth--;
            }

            this.Pooled = new Size(pooledHeight, pooledWidth);
        }

        public override int MinTopBlobs
        {
            get { return 1; }
        }

        protected int _maxTopBlobs;
        public override int MaxTopBlobs
        {
            get { return _maxTopBlobs; }
        }

        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        protected Size Pooled 
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get; 
            set; 
        }
    }
}
