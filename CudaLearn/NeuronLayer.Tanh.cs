using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
   public class TanhLayerConfiguration : LayerConfiguration
    {
        public TanhLayerConfiguration() : base (LayerType.Tahn)
        {}
    }

    /// <summary>
    /// TanhLayer
    ///       Hyperbolic tangent non-linearity, popular in auto-encoders.
    ///     
    /// y = 1. * (exp(2x) - 1) / (exp(2x) + 1)
    /// 
    /// y' = 1 - ( (exp(2x) - 1) / (exp(2x) + 1) ) ^ 2
    /// </summary>
    public class TanhLayer : NeuronLayer<TanhLayerConfiguration>
    {
        public TanhLayer()
            : this(new TanhLayerConfiguration())
        { }

        public TanhLayer(TanhLayerConfiguration param)
            : base(param)
        { }

        protected override float ForwardCpu(IList<Blob> bottom, IList<Blob> top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            bottomData.MapIndexed((i, v) => {
                var exp2x = (float) Math.Exp(2 * v);
                return (exp2x - 1) / (exp2x + 1);
            }, topData, Zeros.Include);

            return 0;
        }

        protected override void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            if ( propagateDown[0] )
            {
                var topData = top[0].Data;
                var topDiff = top[0].Diff;

                var bottomDiff = bottom[0].Diff;                      

                topData.MapIndexed((i, v) => topDiff[i] * (1 - v * v ), bottomDiff, Zeros.Include);
            }
        }
    }
}

