using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class DropoutLayerConfiguration : LayerConfiguration
    {
        public DropoutLayerConfiguration()
            : this(0.0d)
        { }

        public DropoutLayerConfiguration(double ratio)
            : base(LayerType.Dropout)
        {
            Guard.That(() => ratio).IsGreaterOrEqualThan(0);
            Guard.That(() => ratio).IsLessOrEqualThan(1);

            this.Ratio = ratio;
        }

        public double Ratio { get; private set; }
    }

    /// <summary>
    /// DropoutLayer
    ///        During training only, sets some portion of x to 0, adjusting the vector magnitude accordingly.
    ///     
    /// mask = bernoulli(1 - threshold)
    /// scale = 1 / (1 - threshold)
    /// y = x * mask * scale
    /// 
    /// y' = mask * scale
    /// 
    /// </summary>
    public class DropoutLayer : NeuronLayer<DropoutLayerConfiguration>
    {
        private Vector<double> mask;

        public DropoutLayer()
            : this(new DropoutLayerConfiguration())
        { }

        public DropoutLayer(DropoutLayerConfiguration param)
            : base(param)
        { }

        internal override double ForwardCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            if ( Phase == PhaseType.Train)
            {
                var ratio = this.Parameters.Ratio;
                var scale = 1f / (1f - ratio);

                var bernoulli = new Bernoulli(1 - ratio);
                mask = Vector<double>.Build.SameAs(bottomData, () => scale * bernoulli.Sample());

                bottomData.PointwiseMultiply(mask, result: topData);
            }
            else
            {
                bottomData.CopyTo(topData);
            }

            return 0;
        }

        internal override void BackwardCpu(CpuTensorScopeCollection top, IList<bool> propagateDown, CpuTensorScopeCollection bottom)
        {
            if (propagateDown[0])
            {
                var bottomDiff = bottom[0].Diff;
                var topDiff = top[0].Diff;

                if ( this.Phase == PhaseType.Train )
                {                
                    topDiff.PointwiseMultiply(mask, result: bottomDiff);
                }
                else
                {
                    topDiff.CopyTo(bottomDiff);
                }                
            }
        }
    }
}
