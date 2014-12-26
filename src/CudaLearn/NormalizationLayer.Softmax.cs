using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Providers.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{

    public class SoftmaxLayerConfiguration : LayerConfiguration
    {
        public SoftmaxLayerConfiguration()
            : base(LayerType.Softmax)
        { }
    }

    /// <summary>
    ///  SoftmaxLayer
    ///     Implements the softmax function ( normalized exponential ).
    /// </summary>
    public class SoftmaxLayer : Layer<SoftmaxLayerConfiguration>
    {
        private Vector<double> cache;
        private Vector<double> scaleVector;

        public SoftmaxLayer()
            : this(new SoftmaxLayerConfiguration())
        { }

        public SoftmaxLayer(SoftmaxLayerConfiguration param)
            : base(param)
        { }

        public override void Setup(TensorCollection bottom, TensorCollection top)
        {
            base.Setup(bottom, top);

            var bottomBlob = bottom.First();
            var topBlob = top.First();
            topBlob.ReshapeAs(bottomBlob);

            this.cache = Vector<double>.Build.Dense(bottom[0].Count / bottom[0].Num);
            this.scaleVector = Vector<double>.Build.Dense(bottomBlob.Num);
        }

        internal override double ForwardCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
        {
            var bottomData = bottom[0].Data;
            var topData = top[0].Data;

            int num = bottom[0].Num;
            int dim = bottom[0].Count / num;

            // Implementation based on http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
            for (int n = 0; n < num; n++ )
            {
                int offset = n * dim;

                double scale = double.NegativeInfinity;
                for ( int i = 0; i < dim; i++ )
                {
                    if (bottomData[offset + i] > scale)
                        scale = bottomData[offset + i];                                                
                }

                // Store the scale value to use when performing the backwards step.
                this.scaleVector[n] = scale;

                double z = 0.0d;
                for ( int i = 0; i < dim; i++ )
                {
                    double value = Math.Exp(bottomData[offset + i] - scale);                    
                    z += value;

                    // Store in the cache to avoid having to calculate this value again. 
                    cache[i] = value;
                }
                    
                for ( int i = 0; i < dim; i++)
                    topData[offset + i] = (cache[i] / z);
            }

            return 0;
        }

        internal override void BackwardCpu(CpuTensorScopeCollection top, IList<bool> propagateDown, CpuTensorScopeCollection bottom)
        {
            var topDiff = top[0].Diff;
            var bottomDiff = bottom[0].Diff;
            var topData = top[0].Data;

            int num = bottom[0].Num;
            int dim = bottom[0].Count / num;

            // Copy gradients to the bottom layer.
            topDiff.CopyTo(bottomDiff);

            for (int n = 0; n < num; n++)
            {
                int offset = n * dim;

                // REMARK: Numerically unstable dot implementation.
                double scale = 0;
                for (int i = 0; i < dim; i++)
                    scale += topDiff[offset + i] * topData[offset + i];

                for (int i = 0; i < dim; i++)
                    bottomDiff[offset + i] = (topDiff[offset + i] - scale) * topData[offset + i];
            }            
        }

        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

    }
}
