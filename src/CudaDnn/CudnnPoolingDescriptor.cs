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
        internal CudnnPoolingDescriptorHandle Handle;  

        internal CudnnPoolingDescriptor(CudnnPoolingDescriptorHandle handle)
        {
            if (handle.Pointer == IntPtr.Zero)
                throw new ArgumentException("The handle pointer is null.", "handle");

            Contract.Ensures(handle.Pointer != IntPtr.Zero);
            Contract.EndContractBlock();

            this.Handle = handle;
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
            Contract.Ensures(this.Handle.Pointer == IntPtr.Zero);

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnDestroyPoolingDescriptor(this.Handle));
            this.Handle.Pointer = IntPtr.Zero;
        }

        private CudnnPoolingDescriptorParameters descriptorParams;

        public CudnnPoolingDescriptorParameters Parameters
        {
            get
            {
                ThrowIfNotInitialized();
                return this.descriptorParams;
            }
        }

        private void ThrowIfNotInitialized()
        {
            if (!IsInitialized)
                throw new InvalidOperationException("Not initialized.");
        }

        public bool IsInitialized
        {
            get { return this.Handle.Pointer != IntPtr.Zero && this.descriptorParams != null; }
        }

        public void SetParameters(CudnnPoolingDescriptorParameters param)
        {
            if (param == null)
                throw new ArgumentNullException("param");

            Contract.EndContractBlock();

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnSetPoolingDescriptor(
                    this.Handle, param.Mode,
                    param.Height, param.Width, 
                    param.HeightStride, param.WidthStride));

            this.descriptorParams = param;
        }
    }

    public class CudnnPoolingDescriptorParameters
    {
        public readonly CudnnPoolingMode Mode;

        public readonly int Height;
        public readonly int Width;
        public readonly int HeightStride;
        public readonly int WidthStride;

        public CudnnPoolingDescriptorParameters(CudnnPoolingMode mode, int windowHeight, int windowWidth, int verticalStride, int horizontalStride)
        {
            if (windowHeight < 1 || windowWidth < 1)
                throw new ArgumentException("At least one of the parameters windowHeight or windowWidth is negative");

            if (verticalStride < 1 || horizontalStride < 1)
                throw new ArgumentException("At least one of the parameters verticalStride or horizontalStride is negative");

            Contract.EndContractBlock();

            this.Mode = mode;

            this.Height = windowHeight;
            this.Width = windowWidth;
            this.HeightStride = verticalStride;
            this.WidthStride = horizontalStride;
        }
    }
}
