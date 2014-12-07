using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class PowerLayerConfiguration : LayerConfiguration
    {
        public double Power { get; private set; }
        public double Scale { get; private set; }
        public double Shift { get; private set; }

        public double DiffScale 
        { 
            get { return Scale * Power; } 
        }

        public PowerLayerConfiguration()
            : this(1.0d, 1.0d, 0.0d)
        { }

        public PowerLayerConfiguration(double power, double scale, double shift)
            : base(LayerType.Power)
        {
            this.Power = power;
            this.Scale = scale;
            this.Shift = shift;
        }        
    }

    /// <summary>
    /// PowerLayer
    ///     
    ///     y = (shift + scale * x) ^ power
    /// 
    ///     y' = scale * power * (shift + scale * x) ^ (power - 1)
    ///        = scale * power * y / (shift + scale * x)
    /// </summary>
    public class PowerLayer : NeuronLayer<PowerLayerConfiguration>
    {
        public PowerLayer()
            : this(new PowerLayerConfiguration())
        { }

        public PowerLayer(PowerLayerConfiguration param)
            : base(param)
        { }

        protected override double ForwardCpu(IList<Blob> bottom, IList<Blob> top)
        {
            var topData = top[0].Data;

            var power = this.Parameters.Power;
            var shift = this.Parameters.Shift;
            var scale = this.Parameters.Scale;

            // Special case where we can ignore the input: scale or power is 0.
            if (this.Parameters.DiffScale == 0f)
            {
                double value = (power == 0f) ? 1.0d : Math.Pow(shift, power);
                topData.Map(v => value, topData, Zeros.Include);

                return 0;
            }

            // TODO Math.Pow with doubles is numerically highly unstable. Consider change everything to doubles or build a more stable version.
            var bottomData = bottom[0].Data;
            if ( power != 1 )
                bottomData.Map(v => Math.Pow((v) * scale + shift, power), topData, Zeros.Include);
            else
                bottomData.Map(v => v * scale + shift, topData, Zeros.Include);

            return 0;
        }

        protected override void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            if (propagateDown[0])
            {
                var bottomDiff = bottom[0].Diff;
                var topDiff = top[0].Diff;

                var diffScale = this.Parameters.DiffScale;
                var power = this.Parameters.Power;

                if (diffScale == 0 || power == 1)
                {                   
                    bottomDiff.Map(v => diffScale, bottomDiff, Zeros.Include);
                }
                else
                {
                    var bottomData = bottom[0].Data;                    

                    var scale = this.Parameters.Scale;
                    var shift = this.Parameters.Shift;

                    // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
                    //               = diff_scale * y / (shift + scale * x)

                    if (power == 2)
                    {
                        // Special case for y = (shift + scale * x)^2
                        //     -> dy/dx = 2 * scale * (shift + scale * x)
                        //              = diff_scale * shift + diff_scale * scale * x
                        if ( shift != 0 )
                            bottomData.Map(v => diffScale * shift + diffScale * scale * v, bottomDiff, Zeros.Include);
                        else
                            bottomData.Map(v => diffScale * scale * v, bottomDiff, Zeros.Include);                        
                    }
                    else if (shift == 0)
                    {
                        // Special case for y = (scale * x)^power
                        //     -> dy/dx = scale * power * (scale * x)^(power - 1)
                        //              = scale * power * (scale * x)^power * (scale * x)^(-1)
                        //              = power * y / x

                        var topData = top[0].Data;
                        bottomData.MapIndexed((i, v) => power * (topData[i] / v), bottomDiff, Zeros.Include);
                    }
                    else
                    {
                        bottomData.CopyTo(bottomDiff);
                        if (scale != 1)
                            bottomDiff.Multiply(scale, result: bottomDiff);

                        var topData = top[0].Data;
                        bottomDiff.MapIndexed((i, v) => topData[i] / ( v + shift ), bottomDiff, Zeros.Include);

                        if (diffScale != 1)
                            bottomDiff.Multiply(diffScale, result: bottomDiff);
                    }
                }

                if (diffScale != 0)
                    topDiff.PointwiseMultiply(bottomDiff, result: bottomDiff);
            }
        }
    }
}
