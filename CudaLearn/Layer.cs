using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public enum LayerType
    {
        None,
        Threshold,
        Relu,
    }

    public class LayerConfiguration
    {
        public LayerType Type { get; private set; }

        public LayerConfiguration( LayerType type )
        {
            this.Type = type;
        }
    }


    public abstract class Layer
    {
        // We assume we always have gpu.
        private bool forwardGpuSupported = true;
        private bool backwardGpuSupported = true;

        protected Layer()
        {
        }

        public void Setup( Blob bottom, Blob top )
        {
            var bottomList = new List<Blob> { bottom };
            var topList = new List<Blob> { top };
            this.Setup( bottomList, topList );
        }

        public virtual void Setup(IList<Blob> bottom, IList<Blob> top)
        {
            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();
   
            CheckBlobCount(bottom, top);
        }

        public float Forward(Blob bottom, Blob top)
        {
            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();

            var bottomList = new List<Blob> { bottom };
            var topList = new List<Blob> { top };

            return this.Forward(bottomList, topList);
        }

        public float Forward(IList<Blob> bottom, IList<Blob> top)
        {
            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();

#if EXHAUSTIVE_DEBUG

            Guard.That(() => bottom).IsTrue(x => !x.Contains(null), "Cannot contain null.");
            Guard.That(() => top).IsTrue(x => !x.Contains(null), "Cannot contain null.");

#endif

            if (forwardGpuSupported)
            {
                try
                {
                    return ForwardGpu(bottom, top);
                }
                catch (NotSupportedException)
                {
                    forwardGpuSupported = false;
                }
            }

            return ForwardCpu(bottom, top);
        }

        public void Backward(Blob top, IList<bool> propagateDown, Blob bottom)
        {
            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();

            var bottomList = new List<Blob> { bottom };
            var topList = new List<Blob> { top };

            this.Backward(bottomList, propagateDown, topList);
        }

        public void Backward(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();
            Guard.That(() => propagateDown).IsNotNull();

#if EXHAUSTIVE_DEBUG

            Guard.That(() => bottom).IsTrue(x => !x.Contains(null), "Cannot contain null.");
            Guard.That(() => top).IsTrue(x => !x.Contains(null), "Cannot contain null.");
#endif

            if (backwardGpuSupported)
            {
                try
                {
                    BackwardGpu(top, propagateDown, bottom);
                    return;
                }
                catch (NotSupportedException)
                {
                    backwardGpuSupported = false;
                }
            }

            BackwardCpu(top, propagateDown, bottom);
        }

        protected abstract float ForwardCpu(IList<Blob> bottom, IList<Blob> top);

        protected virtual float ForwardGpu(IList<Blob> bottom, IList<Blob> top)
        {
            throw new NotSupportedException();
        }


        protected virtual void BackwardCpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            throw new NotSupportedException();
        }

        protected virtual void BackwardGpu(IList<Blob> top, IList<bool> propagateDown, IList<Blob> bottom)
        {
            throw new NotSupportedException();
        }

        protected virtual void CheckBlobCount(IList<Blob> bottom, IList<Blob> top)
        {
            // Bottom layer
            if (ExactNumBottomBlobs >= 0)
            {
                if (ExactNumBottomBlobs != bottom.Count)
                    throw new ArgumentException(string.Format("{0} Layer takes {1} bottom blob(s) as input.", this.GetType().Name, this.ExactNumBottomBlobs));
            }

            if (MinBottomBlobs >= 0)
            {
                if (bottom.Count <= MinBottomBlobs)
                    throw new ArgumentOutOfRangeException(string.Format("{0} Layer takes at least {1} bottom blob(s) as input.", this.GetType().Name, this.MinBottomBlobs));
            }

            if (MaxBottomBlobs >= 0)
            {
                if (bottom.Count >= MaxBottomBlobs)
                    throw new ArgumentOutOfRangeException(string.Format("{0} Layer takes at most {1} bottom blob(s) as input.", this.GetType().Name, this.MaxBottomBlobs));
            }

            // Top layer
            if (ExactNumTopBlobs >= 0)
            {
                if (ExactNumTopBlobs != top.Count)
                    throw new ArgumentException(string.Format("{0} Layer takes {1} top blob(s) as input.", this.GetType().Name, this.ExactNumTopBlobs));
            }

            if (MinTopBlobs >= 0)
            {
                if (top.Count <= MinTopBlobs)
                    throw new ArgumentOutOfRangeException(string.Format("{0} Layer takes at least {1} top blob(s) as input.", this.GetType().Name, this.MinTopBlobs));
            }

            if (MaxTopBlobs >= 0)
            {
                if (top.Count >= MaxTopBlobs)
                    throw new ArgumentOutOfRangeException(string.Format("{0} Layer takes at most {1} top blob(s) as input.", this.GetType().Name, this.MaxTopBlobs));
            }
        }


        // These properties can be overwritten to declare that this layer type expects
        // a certain number of blobs as input and output.

        // ExactNum{Bottom,Top}Blobs return a non-negative number to require an exact
        // number of bottom/top blobs; the Min/Max versions return a non-negative
        // number to require a minimum and/or maximum number of blobs.
        // If Exact is specified, neither Min nor Max should be specified, and vice
        // versa.  

        // These methods may not rely on Setup having been called.

        public virtual int ExactNumBottomBlobs
        {
            get { return -1; }
        }
        public virtual int MinBottomBlobs
        {
            get { return -1; }
        }
        public virtual int MaxBottomBlobs
        {
            get { return -1; }
        }
        public virtual int ExactNumTopBlobs
        {
            get { return -1; }
        }
        public virtual int MinTopBlobs
        {
            get { return -1; }
        }
        public virtual int MaxTopBlobs
        {
            get { return -1; }
        }


        // EqualNumBottomTopBlobs should return true for layers requiring an equal
        // number of bottom and top blobs.
        public virtual bool EqualNumBottomTopBlobs
        {
            get { return false; }
        }

        // Declare for each bottom blob whether to allow force_backward -- that is,
        // if AllowForceBackward(i) == false, we will ignore the force_backward
        // setting and backpropagate to blob i only if it needs gradient information
        // (as is done when force_backward == false).
        protected virtual bool AllowForceBackward(int bottom_index)
        {
            return true;
        }

        public abstract LayerType Type { get; }
    }

    public abstract class Layer<TConfiguration> : Layer
        where TConfiguration : LayerConfiguration
    {
        public TConfiguration Parameters { get; private set; }

        private LayerType type_ = LayerType.None;

        public override LayerType Type
        {
            get { return this.type_; }
        }

        protected Layer(TConfiguration param)
        {
            Guard.That(() => param).IsNotNull();

            this.Parameters = param;
            this.type_ = param.Type;
        }
    }
}
