using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public abstract class LossLayer<TConfiguration> : Layer<TConfiguration> where TConfiguration : LayerConfiguration
    {
        protected LossLayer(TConfiguration param)
            : base(param)
        {}

        public virtual void PostSetup(IList<Tensor> bottom, IList<Tensor> top) { }

        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        public override int MaxTopBlobs
        {
            get { return 1; }
        }

        protected override bool AllowForceBackward(int bottomIndex)
        {
            return bottomIndex != 1;
        }
    }
}
