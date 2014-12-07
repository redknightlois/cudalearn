using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class UniformFillerConfiguration : FillerConfiguration
    {
        public UniformFillerConfiguration() : this(0.0d, 1.0d) { }

        public UniformFillerConfiguration(double min, double max)
            : base(FillerType.Uniform)
        {
            this.Min = min;
            this.Max = max;
        }

        public double Min { get; set; }
        public double Max { get; set; }
    }

    public class UniformFiller : Filler<UniformFillerConfiguration>
    {
        public UniformFiller()
            : this(new UniformFillerConfiguration())
        { }

        public UniformFiller(UniformFillerConfiguration param)
            : base(param)
        { }

        public UniformFiller(double min, double max) : this ( new UniformFillerConfiguration( min, max ))
        {}

        public override void Fill(Blob blob)
        {
            var data = blob.Data;

            var distribution = new ContinuousUniform(this.Parameters.Min, this.Parameters.Max);
            data.MapInplace(x => distribution.Sample(), Zeros.Include);
        }
    }
}
