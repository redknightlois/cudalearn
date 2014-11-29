using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class DirectConvolutionalLayerConfiguration : ConvolutionalLayerConfiguration
    {
        public DirectConvolutionalLayerConfiguration(int kernelSize, int stride = 1, int padding = 0)
            : this(new Size(kernelSize, kernelSize), new Size(stride, stride), new Size(padding, padding))
        { }

        public DirectConvolutionalLayerConfiguration(Size kernel, Size stride, Size padding = default(Size))
            : base(LayerType.DirectConvolutional2d)
        {
            this.Kernel = kernel;
            this.Stride = stride;
            this.Padding = padding;
        }

        public DirectConvolutionalLayerConfiguration()
            : base(LayerType.DirectConvolutional2d)
        { }
    }

    public class DirectConvolutional2dLayer : ConvolutionalLayer<DirectConvolutionalLayerConfiguration>
    {
        private Blob bias;        
        private Blob weights;
        private Blob biasMultiplier;
        private Blob imageBuffer;

        private int m, n, k;

        public DirectConvolutional2dLayer(DirectConvolutionalLayerConfiguration param)
            : base(param)
        { }

        public override void Setup(IList<Blob> bottom, IList<Blob> top)
        {
            base.Setup(bottom, top);

            Guard.That(() => this.Parameters.Kernel.Width).IsGreaterThan(0);
            Guard.That(() => this.Parameters.Kernel.Height).IsGreaterThan(0);
            Guard.That(() => this.Parameters.Kernel.Depth).IsEqual(0);

            Guard.That(() => this.Parameters.NumberOfOutputs).IsGreaterThan(0);

            int num = bottom[0].Num;
            int channels = bottom[0].Channels;
            int height = bottom[0].Height;
            int width = bottom[0].Width;

            foreach ( var bottomBlob in bottom )
            {
                Guard.That(() => bottomBlob.Num).IsEqual(num);
                Guard.That(() => bottomBlob.Channels).IsEqual(channels);
                Guard.That(() => bottomBlob.Height).IsEqual(height);
                Guard.That(() => bottomBlob.Width).IsEqual(width);
            }

            Size padding = this.Parameters.Padding;
            Size stride = this.Parameters.Stride;
            Size kernel = this.Parameters.Kernel;

            if (this.IsScaleKernel)
                throw new NotSupportedException("1x1 kernels are not supported yet.");

            // Number of output should be multiples of group.
            Guard.That(() => this.Parameters.NumberOfOutputs % this.Parameters.Groups).IsEqual(0);
            // Number of channels should be multiples of group.
            Guard.That(() => channels % this.Parameters.Groups).IsEqual(0);

            // We are going to work 1 image at a time to avoid overly large memory usage.
            int outputHeight = (height + 2 * padding.Height - kernel.Height) / stride.Height + 1;
            int outputWidth = (width + 2 * padding.Width - kernel.Width) / stride.Width + 1;
            this.imageBuffer = new Blob( 1, channels * kernel.Height * kernel.Width, outputHeight, outputWidth);

            // Figure out the dimensions for individual multiplications.
            this.m = this.Parameters.NumberOfOutputs / this.Parameters.Groups;
            this.k = channels * kernel.Height * kernel.Width / this.Parameters.Groups;
            this.n = outputHeight * outputWidth;

            // Resize the output
            foreach ( var topBlob in top)
                topBlob.Reshape( num, this.Parameters.NumberOfOutputs, outputHeight, outputWidth);

            // Initialize the weight
            this.weights = new Blob(this.Parameters.NumberOfOutputs, channels / this.Parameters.Groups, this.Parameters.Kernel.Height, this.Parameters.Kernel.Width);
            var weightsFiller = FillerFactory.Create(this.Parameters.WeightsFiller);
            weightsFiller.Fill(this.weights);            
            this.SetPropagateDownForParameter(0, true);

            if ( this.Parameters.UseBias )
            {
                // Initialize the bias
                this.bias = new Blob(1, 1, 1, this.Parameters.NumberOfOutputs);
                var biasFiller = FillerFactory.Create(this.Parameters.BiasFiller);
                biasFiller.Fill(this.bias);

                this.SetPropagateDownForParameter(1, true);
            }

            this.biasMultiplier = new Blob(1, 1, 1, this.n);
            this.biasMultiplier.InitializeWith(1, 0);
        }

        protected override float ForwardCpu(IList<Blob> bottom, IList<Blob> top)
        {
            var bottomBlob = bottom.First();

            int num = bottomBlob.Num;
            int channels = bottomBlob.Channels;
            int height = bottomBlob.Height;
            int width = bottomBlob.Width;

            var topData = top.First().Data;
            var bottomData = (DenseVector)bottomBlob.Data;
            var columnData = this.imageBuffer.Data;
            var weightData = this.weights.Data;

            int weightOffset = this.m * this.k;
            int columnOffset = this.k * this.n;
            int topOffset = this.m * this.n;            

            for (int i = 0; i < num; i++ )
            {
                bottomBlob.Image2Column(i, target: imageBuffer);

                for (int g = 0; g < this.Parameters.Groups; g++ )
                {

                }

                if (Parameters.UseBias)
                {

                }
            }

            throw new NotImplementedException();
        }

        protected override void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {




            throw new NotImplementedException();
        }
    }
}
