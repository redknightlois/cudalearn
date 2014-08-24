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

    public static class FillerFactory
    {
        public static Filler Create(FillerConfiguration configuration)
        {
            switch ( configuration.Type )
            {
                case FillerType.Constant: return new ConstantFiller((ConstantFillerConfiguration)configuration);
                case FillerType.Gaussian: return new GaussianFiller((GaussianFillerConfiguration)configuration);
                case FillerType.PositiveUnitball: return new PositiveUnitballFiller((PositiveUnitballFillerConfiguration)configuration);
                case FillerType.Uniform: return new UniformFiller((UniformFillerConfiguration)configuration);
                case FillerType.Xavier: return new XavierFiller((XavierFillerConfiguration)configuration);
                default: throw new NotSupportedException(string.Format("Filler type {0} is not supported by the FillerFactory", configuration.Type));
            }
        }
    }
}
