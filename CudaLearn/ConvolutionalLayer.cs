using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class ConvolutionalLayerConfiguration : LayerConfiguration
    {
        protected ConvolutionalLayerConfiguration(LayerType type)
            : base(type)
        {
            this.Kernel = new Size(2, 2);
            this.Stride = new Size(1, 1);
            this.Padding = new Size();
        }

        public Size Kernel { get; set; }

        public Size Stride { get; set; }

        public Size Padding { get; set; }


        public int NumberOfOutputs { get; set; }

        public int Groups { get; set; }

        public bool UseBias { get; set; }

        public FillerConfiguration BiasFiller { get; set; }

        public FillerConfiguration WeightsFiller { get; set; }

    }

    public abstract class ConvolutionalLayer<TConfiguration> : Layer<TConfiguration> where TConfiguration : ConvolutionalLayerConfiguration
    {
        protected ConvolutionalLayer(TConfiguration param)
            : base(param)
        {
        }

        public override void Setup(IList<Blob> bottom, IList<Blob> top)
        {
            base.Setup(bottom, top);
        }

        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        public override int MinTopBlobs
        {
            get { return 1; }
        }

        public override bool EqualNumBottomTopBlobs
        {
            get { return true; }
        }

        public virtual bool IsScaleKernel
        {
            get { return this.Parameters.Kernel.Width == 1 && this.Parameters.Kernel.Height == 1; }
        }
    }
}
