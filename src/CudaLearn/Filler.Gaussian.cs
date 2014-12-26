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
        public GaussianFillerConfiguration() : this(0.0d, 1.0d) { }

        public GaussianFillerConfiguration(double mean, double stdDev)
            : base(FillerType.Gaussian)
        {
            this.Mean = mean;
            this.Std = stdDev;
        }

        public double Mean { get; set; }
        public double Std { get; set; }

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

        public override void Fill(Tensor blob)
        {
            using (var @cpuBlob = blob.OnCpu())
            {
                var data = @cpuBlob.Data;

                var distribution = new Normal(this.Parameters.Mean, this.Parameters.Std);
                data.MapInplace(x => distribution.Sample(), Zeros.Include);

                if (this.Parameters.IsSparse)
                {
                    Guard.That(() => blob.Num).Equals(1);
                    Guard.That(() => blob.Channels).Equals(1);

                    int numberOfInputs = blob.Height;
                    double nonZeroProbability = 1.0d / numberOfInputs;

                    var bernoulli = new Bernoulli(nonZeroProbability);
                    var mask = Vector<double>.Build.SameAs(data, () => bernoulli.Sample());

                    data.PointwiseMultiply(mask, result: data);
                }
            }
        }
    }
}
