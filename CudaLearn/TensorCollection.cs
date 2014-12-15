using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class TensorCollection : List<Tensor>
    {
        public TensorCollection()
            : base()
        { }
        public TensorCollection(int capacity)
            : base(capacity)
        { }
        public TensorCollection(IEnumerable<Tensor> list) : base ( list )
        { }


        public CpuTensorScopeCollection OnCpu ()
        {
            return new CpuTensorScopeCollection(this.Select(x => x.OnCpu())); 
        }

        public GpuTensorScopeCollection OnGpu ()
        {
            return new GpuTensorScopeCollection(this.Select(x => x.OnGpu())); 
        }

        public static implicit operator TensorCollection(Tensor[] tensors)
        {
            Contract.Requires(tensors != null);

            return new TensorCollection(tensors);
        }
    }

    public class CpuTensorScopeCollection : List<CpuTensorScope>, IDisposable
    {
        public CpuTensorScopeCollection()
            : base()
        { }
        public CpuTensorScopeCollection(int capacity)
            : base(capacity)
        { }

        public CpuTensorScopeCollection(CpuTensorScope item)
            : base(new[] { item })
        { }

        public CpuTensorScopeCollection(IEnumerable<CpuTensorScope> list)
            : base(list)
        { }

        ~CpuTensorScopeCollection()
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
                foreach (var item in this)
                    item.Dispose();

                // Check if this is OK.
                this.Clear();
            }           
        }
    }

    public class GpuTensorScopeCollection : List<GpuTensorScope>, IDisposable
    {
        public GpuTensorScopeCollection()
            : base()
        { }
        public GpuTensorScopeCollection(int capacity)
            : base(capacity)
        { }

        public GpuTensorScopeCollection(GpuTensorScope item) : base (new[] { item })
        { }

        public GpuTensorScopeCollection(IEnumerable<GpuTensorScope> list)
            : base(list)
        { }

        ~GpuTensorScopeCollection()
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
                foreach (var item in this)
                    item.Dispose();

                // Check if this is OK.
                this.Clear();
            }
        }
    }
}
