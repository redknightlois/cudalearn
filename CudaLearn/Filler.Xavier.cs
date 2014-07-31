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

        public override void Fill(Blob blob)
        {
            Guard.That(() => blob.Count).IsPositive();

            var data = blob.Data;

            int fanIn = blob.Count / blob.Num;
            float scale = (float) Math.Sqrt(3 / fanIn);

            var distribution = new ContinuousUniform(-scale, scale);
            data.MapInplace(x => (float)distribution.Sample(), Zeros.Include);
        }
    }
}
