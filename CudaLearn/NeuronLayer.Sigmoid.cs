using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class SigmoidLayerConfiguration : LayerConfiguration
    {
        public SigmoidLayerConfiguration()
            : base(LayerType.Sigmoid)
        { }
    }

    /// <summary>
    /// SigmoidLayer
    /// 
    ///   Sigmoid function non-linearity, a classic choice in neural networks. 
    ///   Note that the gradient vanishes as the values move away from 0.
    ///   The ReLULayer is often a better choice for this reason.
    ///     
    ///     y = 1. / (1 + exp(-x)) 
    /// 
    ///     y ' = exp(x) / (1 + exp(x))^2
    ///     or
    ///     y' = y * (1 - y)
    /// </summary>
    public class SigmoidLayer : NeuronLayer<SigmoidLayerConfiguration>
    {

        private const float Threshold = 50.0f;

        public SigmoidLayer()
            : this(new SigmoidLayerConfiguration())
        { }

        public SigmoidLayer(SigmoidLayerConfiguration param)
            : base(param)
        { }

        protected override float ForwardCpu(IList<Blob> bottom, IList<Blob> top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            bottomData.MapIndexed((i, v) => 1.0f / (1.0f + (float)Math.Exp(-v)), topData, Zeros.Include);

            return 0;
        }

        protected override void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            if ( propagateDown[0])
            {
                var topData = top[0].Data;
                var topDiff = top[0].Diff;
                var bottomDiff = bottom[0].Diff;

                topData.MapIndexed((i, v) => topDiff[i] * v * (1.0f - v), bottomDiff, Zeros.Include);
            }
        }
    }
}
