using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{

    public class ThresholdLayerConfiguration : LayerConfiguration
    {
        public ThresholdLayerConfiguration()
            : this(0.0d)
        { }

        public ThresholdLayerConfiguration(double threshold)
            : base(LayerType.Threshold)
        {
            this.Threshold = threshold;
        }

        public double Threshold { get; set; }
    }

    /// <summary>
    /// ThresholdLayer
    ///     Outputs 1 if value in input is above threshold, 0 otherwise.
    ///     The default threshold = 0, which means positive values would become 1 and 
    ///     negative or 0, would become 0
    ///     
    /// y = 1 if x greater than Threshold
    /// y = 0 if x less or equal than Threshold
    /// 
    /// y' = don't differentiable
    /// </summary>
    public class ThresholdLayer : NeuronLayer<ThresholdLayerConfiguration>
    {
        public ThresholdLayer()
            : this(new ThresholdLayerConfiguration())
        { }

        public ThresholdLayer(ThresholdLayerConfiguration param)
            : base(param)
        { }

        internal override double ForwardCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            var threshold = this.Parameters.Threshold;

            int count = bottom[0].Count;
            bottomData.MapIndexed((i, v) => (v > threshold) ? 1.0d : 0.0d, topData, Zeros.Include);

            return 0;
        }
    }
}
