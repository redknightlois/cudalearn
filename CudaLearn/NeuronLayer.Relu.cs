using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    /// Allow non-zero slope for negative inputs to speed up optimization
    /// Described in:
    /// Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities
    /// improve neural network acoustic models. In ICML Workshop on Deep Learning
    /// for Audio, Speech, and Language Processing.
    public class ReluLayerConfiguration : LayerConfiguration
    {
        public ReluLayerConfiguration()
            : this(0.0f)
        { }

        public ReluLayerConfiguration(float negativeSlope) : base ( LayerType.Relu )
        {
            Guard.That(() => negativeSlope).IsGreaterOrEqualThan(0);

            this.NegativeSlope = negativeSlope;
        }

        public float NegativeSlope { get; private set; }
    }

    /// <summary>
    /// ReluLayer
    ///       Rectified Linear Unit non-linearity.
    ///       The simple max is fast to compute, and the function does not saturate.
    ///     
    /// y = max(0, x) if x is greater or equal to 0.
    /// y = negative_slope * x if x is less than 0
    /// 
    /// y' = 0 if x is less or equal than 0
    /// y' = 1 if x greater than 0
    /// 
    /// </summary>
    public class ReluLayer : NeuronLayer<ReluLayerConfiguration>
    {
        public ReluLayer()
            : this(new ReluLayerConfiguration())
        { }

        public ReluLayer(ReluLayerConfiguration param)
            : base(param)
        { }

        protected override float ForwardCpu(IList<Blob> bottom, IList<Blob> top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            var slope = this.Parameters.NegativeSlope;

            int count = bottom[0].Count;
            bottomData.MapIndexed((i, v) => v > 0 ? v : v * slope, topData, Zeros.Include);

            return 0;
        }

        protected override void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            if ( propagateDown[0] )
            {
                var bottomData = bottom[0].Data;
                var bottomDiff = bottom[0].Diff;      
                var topDiff = top[0].Diff;

                int count = bottom[0].Count;

                var slope = this.Parameters.NegativeSlope;

                bottomData.MapIndexed((i, v) => topDiff[i] * ( v > 0.0f ? 1.0f : 0.0f ) + slope * (v <= 0 ? 1.0f : 0.0f), bottomDiff, Zeros.Include);
            }
        }
    }
}
