using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class SoftmaxLossLayerConfiguration : LayerConfiguration
    {
        public SoftmaxLossLayerConfiguration()
            : base(LayerType.SoftmaxLoss)
        { }
    }

    /// <summary>
    ///  SoftmaxLossLayer
    ///     Implements softmax and computes the loss.
    ///     
    ///     It is preferred over separate softmax + multinomial logistic loss
    ///     layers due to more numerically stable gradients.
    ///     
    ///     In test, this layer could be replaced by simple softmax layer.
    /// </summary>
    public class SoftmaxLossLayer : LossLayer<SoftmaxLossLayerConfiguration>
    {
        private readonly SoftmaxLayer softmaxLayer = new SoftmaxLayer();
        private readonly Blob probability = new Blob();

        public SoftmaxLossLayer()
            : this(new SoftmaxLossLayerConfiguration())
        { }

        public SoftmaxLossLayer(SoftmaxLossLayerConfiguration param)
            : base(param)
        { }

        public override void Setup(IList<Blob> bottom, IList<Blob> top)
        {
            base.Setup(bottom, top);
            softmaxLayer.Setup(bottom, new[] { probability } );

            // Softmax loss ( averaged across batch )
            if (top.Count >= 1)
                top[0].Reshape(1, 1, 1, 1);

            // Also adds the softmax function output.
            if ( top.Count == 2 )
                top[1].Reshape(bottom[0].Num, bottom[0].Channels, bottom[0].Height, bottom[0].Width);
        }

        protected override double ForwardCpu(IList<Blob> bottom, IList<Blob> top)
        {
            // The forward pass computes the softmax prob values.
            softmaxLayer.Forward(bottom, new[] { probability });

            var probabilityData = probability.Data;
            var labels = bottom[1].Data;

            int num = bottom[0].Num;
            int dim = bottom[0].Count / num;

            double loss = 0;
            for (int i = 0; i < num; i++ )
                loss -= Math.Log(Math.Max(probabilityData[i * dim + (int)labels[i]], double.Epsilon));

            loss = loss / num;

            if (top.Count >= 1)
                top[0].Data[0] = loss;

            if (top.Count == 2)
                top[1].ShareData(probability);

            return loss;
        }

        protected override void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            if (propagateDown[1])
                throw new NotSupportedException("SoftmaxLossLayer cannot back-propagate to label inputs.");

            if (propagateDown[0])
            {
                var bottomDiff = bottom[0].Diff;
                var labels = bottom[1].Data;

                int num = bottom[0].Num;
                int dim = bottom[0].Count / num;

                for (int i = 0; i < num; i++)
                    bottomDiff[i * dim + (int)labels[i]] -= 1;

                // Scale down gradient
                double scale = 1f / num;
                bottomDiff.Map(v => v * scale, Zeros.Include);
            }
        }
    }
}
