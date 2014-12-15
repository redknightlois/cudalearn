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
    public class XavierFillerConfiguration : FillerConfiguration
    {
        public XavierFillerConfiguration()
            : base(FillerType.Xavier)
        {}
    }

    public class XavierFiller : Filler<XavierFillerConfiguration>
    {
        public XavierFiller() : this ( new XavierFillerConfiguration() )
        {}

        public XavierFiller(XavierFillerConfiguration param)
            : base(param)
        { }

        public override void Fill(Tensor blob)
        {
            Guard.That(() => blob.Count).IsPositive();

            using (var @cpuBlob = blob.OnCpu())
            {
                var data = @cpuBlob.Data;

                int fanIn = blob.Count / blob.Num;
                double scale = Math.Sqrt(3 / fanIn);

                var distribution = new ContinuousUniform(-scale, scale);
                data.MapInplace(x => distribution.Sample(), Zeros.Include);
            }
        }
    }
}
