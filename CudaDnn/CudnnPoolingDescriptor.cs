using CudaDnn.Impl;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using System.Text;
using System.Threading.Tasks;

namespace CudaDnn
{
    public sealed class CudnnPoolingDescriptor : CriticalFinalizerObject, IDisposable
    {
        internal CudnnPoolingDescriptorHandle Value;

        public bool IsValid
        {
            get { return this.Value.Pointer != IntPtr.Zero; }
        }

        internal CudnnPoolingDescriptor(CudnnPoolingDescriptorHandle handle)
        {
            if (handle.Pointer == IntPtr.Zero)
                throw new ArgumentException("The handle pointer is null.", "handle");

            Contract.EndContractBlock();

            this.Value = handle;
        }

        ~CudnnPoolingDescriptor()
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

        private void DisposeManaged() { }


        private void DisposeNative()
        {
            if (this.Value.Pointer == IntPtr.Zero)
                throw new InvalidOperationException("The handle pointer is null.");

            Contract.Ensures(this.Value.Pointer == IntPtr.Zero);
            Contract.EndContractBlock();

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnDestroyPoolingDescriptor(this.Value));
            this.Value.Pointer = IntPtr.Zero;
        }

    }
}
