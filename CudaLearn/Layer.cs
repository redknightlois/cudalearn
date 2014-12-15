using Seterlund.CodeGuard;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public enum LayerType
    {
        None,
        Threshold,
        Relu,
        Bnll,
        Sigmoid,
        Dropout,
        Power,
        Tahn,
        EuclideanLoss,
        SoftmaxLoss,
        Softmax,
        AveragePooling,
        MaxPooling,
        StocasticPooling,
        DirectConvolutional2d,
    }

    public class LayerConfiguration
    {
        public LayerType Type { get; private set; }

        public LayerConfiguration( LayerType type )
        {
            this.Type = type;
        }
    }


    public abstract class Layer : CriticalFinalizerObject, IDisposable
    {
        // We assume we always have GPU.
        private bool forwardGpuSupported = true;
        private bool backwardGpuSupported = true;

        protected Layer()
        {
        }

        ~Layer()
        {
            // Finalizer calls Dispose(false)
            Dispose(false);
        }

        // Dispose() calls Dispose(true)
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // The bulk of the clean-up code is implemented in Dispose(bool)
        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                // free managed resources
                DisposeManaged();
            }

            // free native resources if there are any.
            DisposeNative();
        }

        protected virtual void DisposeManaged() { }

        protected virtual void DisposeNative() { }



        public void Setup( Tensor bottom, Tensor top )
        {
            Contract.Requires(bottom != null);
            Contract.Requires(top != null);

            var bottomList = new TensorCollection { bottom };
            var topList = new TensorCollection { top };
            this.Setup( bottomList, topList );
        }

        public virtual void Setup(TensorCollection bottom, TensorCollection top)
        {
            Contract.Requires(bottom != null);
            Contract.Requires(top != null);

            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();
   
            CheckBlobCount(bottom, top);
        }

        public double Forward(Tensor bottom, Tensor top)
        {
            Contract.Requires(bottom != null);
            Contract.Requires(top != null);

            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();

            var bottomList = new TensorCollection { bottom };
            var topList = new TensorCollection { top };

            return this.Forward(bottomList, topList);
        }

        public double Forward(TensorCollection bottom, TensorCollection top)
        {
            Contract.Requires(bottom != null);
            Contract.Requires(top != null);
            Contract.ForAll<Tensor>(bottom, x => x != null);
            Contract.ForAll<Tensor>(top, x => x != null);

            // TODO Fail if not initialized.
            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();

#if EXHAUSTIVE_DEBUG

            Guard.That(() => bottom).IsTrue(x => !x.Contains(null), "Cannot contain null.");
            Guard.That(() => top).IsTrue(x => !x.Contains(null), "Cannot contain null.");

#endif
            switch (Context.Instance.Mode)
            {
                case ExecutionModeType.Automatic:
                    {
                        if (forwardGpuSupported)
                        {
                            using (var bottomGpu = bottom.OnGpu())
                            using (var topGpu = top.OnGpu())
                            {
                                try
                                {
                                    return ForwardGpu(bottomGpu, topGpu);
                                }
                                catch (NotSupportedException)
                                {
                                    forwardGpuSupported = false;
                                }
                            }
                        }

                        using (var bottomCpu = bottom.OnCpu())
                        using (var topCpu = top.OnCpu())
                        {
                            return ForwardCpu(bottomCpu, topCpu);
                        }
                    }
                case ExecutionModeType.Gpu:
                    {
                        using (var bottomGpu = bottom.OnGpu())
                        using (var topGpu = top.OnGpu())
                        {
                            return ForwardGpu(bottomGpu, topGpu);
                        }
                    }
                case ExecutionModeType.Cpu:
                    {
                        using (var bottomCpu = bottom.OnCpu())
                        using (var topCpu = top.OnCpu())
                        {
                            return ForwardCpu(bottomCpu, topCpu);
                        }
                    }
                default: throw new NotSupportedException(string.Format("Mode of operation '{0}' not support", Context.Instance.Mode.ToString()));
            }
        }        

        public void Backward(Tensor top, IList<bool> propagateDown, Tensor bottom)
        {
            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();

            this.Backward(new TensorCollection { top }, propagateDown, new TensorCollection { bottom });
        }

        public void Backward(TensorCollection top, IList<bool> propagateDown, TensorCollection bottom)
        {
            Contract.Requires(bottom != null);
            Contract.Requires(top != null);
            Contract.Requires(propagateDown != null);
            Contract.ForAll<Tensor>(bottom, x => x != null);
            Contract.ForAll<Tensor>(top, x => x != null);            

            // TODO Fail if not initialized.

            Guard.That(() => bottom).IsNotNull();
            Guard.That(() => top).IsNotNull();
            Guard.That(() => propagateDown).IsNotNull();

#if EXHAUSTIVE_DEBUG

            Guard.That(() => bottom).IsTrue(x => !x.Contains(null), "Cannot contain null.");
            Guard.That(() => top).IsTrue(x => !x.Contains(null), "Cannot contain null.");
#endif

            switch (Context.Instance.Mode)
            {
                case ExecutionModeType.Automatic:
                    {
                        if (backwardGpuSupported)
                        {
                            using (var bottomGpu = bottom.OnGpu())
                            using (var topGpu = top.OnGpu())
                            {
                                try
                                {
                                    BackwardGpu(topGpu, propagateDown, bottomGpu);
                                }
                                catch (NotSupportedException)
                                {
                                    backwardGpuSupported = false;
                                }
                            }
                        }
                        else
                        {
                            using (var bottomCpu = bottom.OnCpu())
                            using (var topCpu = top.OnCpu())
                            {
                                BackwardCpu(topCpu, propagateDown, bottomCpu);
                            }                            
                        }
                       
                        return;
                    }
                case ExecutionModeType.Gpu:
                    {
                        using (var bottomGpu = bottom.OnGpu())
                        using (var topGpu = top.OnGpu())
                        {
                            BackwardGpu(topGpu, propagateDown, bottomGpu);
                        }
                        return;
                    }
                case ExecutionModeType.Cpu:
                    {
                        using (var bottomCpu = bottom.OnCpu())
                        using (var topCpu = top.OnCpu())
                        {
                            BackwardCpu(topCpu, propagateDown, bottomCpu);
                        }                            
                        return;
                    }
                default: throw new NotSupportedException(string.Format("Mode of operation '{0}' not support", Context.Instance.Mode.ToString()));
            }
        }

        internal abstract double ForwardCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top);

        internal virtual double ForwardGpu(GpuTensorScopeCollection bottom, GpuTensorScopeCollection top)
        {
            throw new NotSupportedException();
        }


        internal virtual void BackwardCpu(CpuTensorScopeCollection top, IList<bool> propagateDown, CpuTensorScopeCollection bottom)
        {
            throw new NotSupportedException();
        }

        internal virtual void BackwardGpu(GpuTensorScopeCollection top, IList<bool> propagateDown, GpuTensorScopeCollection bottom)
        {
            throw new NotSupportedException();
        }

        protected virtual void CheckBlobCount(TensorCollection bottom, TensorCollection top)
        {
            // Bottom layer
            if (ExactNumBottomBlobs >= 0)
            {
                if (ExactNumBottomBlobs != bottom.Count)
                    throw new ArgumentException(string.Format("{0} Layer takes {1} bottom blob(s) as input.", this.GetType().Name, this.ExactNumBottomBlobs));
            }

            if (MinBottomBlobs >= 0)
            {
                if (bottom.Count < MinBottomBlobs)
                    throw new ArgumentOutOfRangeException(string.Format("{0} Layer takes at least {1} bottom blob(s) as input.", this.GetType().Name, this.MinBottomBlobs));
            }

            if (MaxBottomBlobs >= 0)
            {
                if (bottom.Count > MaxBottomBlobs)
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
                if (top.Count < MinTopBlobs)
                    throw new ArgumentOutOfRangeException(string.Format("{0} Layer takes at least {1} top blob(s) as input.", this.GetType().Name, this.MinTopBlobs));
            }

            if (MaxTopBlobs >= 0)
            {
                if (top.Count > MaxTopBlobs)
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
        // setting and back-propagate to blob i only if it needs gradient information
        // (as is done when force_backward == false).
        protected virtual bool AllowForceBackward(int bottomIndex)
        {
            return true;
        }

        public abstract LayerType Type { get; }

        public PhaseType Phase { get { return Context.Instance.Phase; } }


        private readonly SortedSet<int> propagateDownList = new SortedSet<int>();

        public void SetPropagateDownForParameter( int parameterId, bool value )
        {
            // When true we add it to the list, when false we remove it.
            if ( value )                 
                propagateDownList.Add(parameterId);
            else
                propagateDownList.Remove(parameterId);
        }

        public bool GetPropagateDownForParameter( int parameterId )
        {
            return propagateDownList.Contains(parameterId);
        }
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
