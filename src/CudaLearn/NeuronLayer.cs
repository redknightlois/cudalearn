using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public abstract class NeuronLayer<TConfiguration> : Layer<TConfiguration> where TConfiguration : LayerConfiguration
    {
        protected NeuronLayer(TConfiguration param)
            : base(param)
        {
        }

        public override void Setup(TensorCollection bottom, TensorCollection top)
        {
            base.Setup(bottom, top);

            // NeuronLayer allows in-place computations. If the computation is not
            // in-place, we will need to initialize the top blob.
            var bottomBlob = bottom.First();
            var topBlob = top.First();
            if ( bottomBlob != topBlob )
                topBlob.ReshapeAs(bottomBlob);
        }

        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }
    }
}
