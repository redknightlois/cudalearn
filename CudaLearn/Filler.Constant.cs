using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class ConstantFillerConfiguration : FillerConfiguration
    {
        public ConstantFillerConfiguration() : this(0.0f) { }

        public ConstantFillerConfiguration(float value)
            : base(FillerType.Constant)
        {
            this.Value = value;
        }

        public float Value { get; set; }
    }

    public class ConstantFiller : Filler<ConstantFillerConfiguration>
    {
        public ConstantFiller()
            : this(new ConstantFillerConfiguration())
        { }

        public ConstantFiller(ConstantFillerConfiguration param)
            : base(param)
        { }

        public override void Fill(Blob blob)
        {
            var data = blob.Data;
            var value = this.Parameters.Value;

            data.MapInplace(x => value, Zeros.Include);
        }
    }
}
