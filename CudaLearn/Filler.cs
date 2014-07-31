using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public enum FillerType
    {
        None,
        Constant,
        Uniform,
        Gaussian,
        PositiveUnitball,
        Xavier,
    }

    public class FillerConfiguration
    {
        public FillerType Type { get; private set; }

        public FillerConfiguration(FillerType type)
        {
            this.Type = type;
        }
    }

    public abstract class Filler
    {
        public abstract FillerType Type { get; }

        public abstract void Fill ( Blob blob );
    }

    public abstract class Filler<TConfiguration> : Filler
    where TConfiguration : FillerConfiguration
    {
        public TConfiguration Parameters { get; private set; }

        private FillerType type_ = FillerType.None;

        public override FillerType Type
        {
            get { return this.type_; }
        }

        protected Filler(TConfiguration param)
        {
            Guard.That(() => param).IsNotNull();

            this.Parameters = param;
            this.type_ = param.Type;
        }
    }
}
