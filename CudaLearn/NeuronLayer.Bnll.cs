using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class BnllLayerConfiguration : LayerConfiguration
    {
        public BnllLayerConfiguration()
            : base(LayerType.Bnll)
        {}
    }

    /// <summary>
    /// BnllLayer
    ///     
    ///     y = x + log(1 + exp(-x))   if x greater than 0
    ///     y = log(1 + exp(x))        if x less or equal than 0
    /// 
    ///     y' = exp(x) / (exp(x) + 1)
    /// </summary>
    public class BnllLayer : NeuronLayer<BnllLayerConfiguration>
    {

        private const float Threshold = 50.0f;

        public BnllLayer()
            : this(new BnllLayerConfiguration())
        { }

        public BnllLayer(BnllLayerConfiguration param)
            : base(param)
        { }

        protected override float ForwardCpu(IList<Blob> bottom, IList<Blob> top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            int count = bottom[0].Count;
            bottomData.MapIndexed((i, v) => (v > 0) ? v + (float)Math.Log(1.0f + Math.Exp(-v)) : (float)Math.Log(1.0f + Math.Exp(v)), topData, Zeros.Include);

            return 0;
        }

        protected override void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            var bottomData = bottom[0].Data;
            var topDiff = top[0].Diff;
            var bottomDiff = bottom[0].Diff;

            int count = bottom[0].Count;
            bottomData.MapIndexed((i, v) => 
                {
                    var expVal = (float) Math.Exp(Math.Min(v, Threshold));
                    return topDiff[i] * expVal / (expVal + 1.0f);
                }, bottomDiff, Zeros.Include);
        }
    }
}
