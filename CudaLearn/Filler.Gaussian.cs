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
    public class GaussianFillerConfiguration : FillerConfiguration
    {
        public GaussianFillerConfiguration() : this(0.0f, 1.0f) { }

        public GaussianFillerConfiguration(float mean, float stdDev)
            : base(FillerType.Gaussian)
        {
            this.Mean = mean;
            this.Std = stdDev;
        }

        public float Mean { get; set; }
        public float Std { get; set; }

        public bool IsSparse { get; set; }
    }

    public class GaussianFiller : Filler<GaussianFillerConfiguration>
    {
        public GaussianFiller()
            : this(new GaussianFillerConfiguration())
        { }

        public GaussianFiller(GaussianFillerConfiguration param)
            : base(param)
        { }

        public override void Fill(Blob blob)
        {
            var data = blob.Data;

            var distribution = new Normal(this.Parameters.Mean, this.Parameters.Std);
            data.MapInplace(x => (float)distribution.Sample(), Zeros.Include);
            
            if ( this.Parameters.IsSparse )
            {
                Guard.That(() => blob.Num).Equals(1);
                Guard.That(() => blob.Channels).Equals(1);

                int numberOfInputs = blob.Height;
                float nonZeroProbability = 1.0f / numberOfInputs;

                var bernoulli = new Bernoulli(nonZeroProbability);
                var mask = Vector<float>.Build.SameAs(blob.Data, () => (float)bernoulli.Sample());

                blob.Data.PointwiseMultiply(mask, result: blob.Data);
            }
        }
    }
}
