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
    public class PositiveUnitballFillerConfiguration : FillerConfiguration
    {
        public PositiveUnitballFillerConfiguration()
            : base(FillerType.PositiveUnitball)
        {}
    }

    public class PositiveUnitballFiller : Filler<PositiveUnitballFillerConfiguration>
    {
        public PositiveUnitballFiller() : this ( new PositiveUnitballFillerConfiguration() )
        { }

        public PositiveUnitballFiller(PositiveUnitballFillerConfiguration param)
            : base(param)
        { }

        public override void Fill(Tensor blob)
        {
            Guard.That(() => blob.Count).IsPositive();

            using (var @cpuBlob = blob.OnCpu())
            {
                var data = @cpuBlob.Data;

                var distribution = new ContinuousUniform(0, 1);
                data.MapInplace(x => distribution.Sample(), Zeros.Include);

                int dim = blob.Count / blob.Num;
                Guard.That(() => dim).IsPositive();

                for (int i = 0; i < blob.Num; i++)
                {
                    double sum = 0.0d;
                    for (int j = 0; j < dim; j++)
                        sum += data[i * dim + j];

                    for (int j = 0; j < dim; j++)
                        data[i * dim + j] /= sum;
                }
            }
        }
    }
}
