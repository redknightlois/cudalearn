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
    public sealed class CudnnFilterDescriptor : CriticalFinalizerObject, IDisposable, ITypeAware
    {
        internal CudnnFilterDescriptorHandle Handle;

        internal CudnnFilterDescriptor(CudnnFilterDescriptorHandle handle)
        {
            if (handle.Pointer == IntPtr.Zero)
                throw new ArgumentException("The handle pointer is null.", "handle");

            Contract.EndContractBlock();

            this.Handle = handle;
        }

        ~CudnnFilterDescriptor()
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

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnDestroyFilterDescriptor(this.Handle));
            this.Handle.Pointer = IntPtr.Zero;
        }

        private CudnnFilterDescriptorParameters descriptorParams;

        public CudnnFilterDescriptorParameters Parameters
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

        public void SetParameters(CudnnFilterDescriptorParameters param)
        {
            if (param == null)
                throw new ArgumentNullException("param");

            Contract.EndContractBlock();

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnSetFilterDescriptor(
                                this.Handle, param.Type,
                                param.Output, param.Input, param.Height, param.Width));

            this.descriptorParams = param;
        }

        public CudnnType Type
        {
            get { return Parameters.Type; }
        }
    }

    public class CudnnFilterDescriptorParameters : ITypeAware
    {
        public CudnnType Type { get; private set; }

        public readonly int Output;
        public readonly int Input;
        public readonly int Height;
        public readonly int Width;

        public CudnnFilterDescriptorParameters ( int output, int input, int h, int w ) 
            : this ( CudnnContext.DefaultType, output, input, h, w )
        {     
        }

        public CudnnFilterDescriptorParameters(CudnnType type, int output, int input, int h, int w)
        {
            if (output < 1 || input < 1 || h < 1 || w < 1)
                throw new ArgumentException("At least one of the parameters output, input, h, w was negative.");

            Contract.EndContractBlock();

            this.Type = type;

            this.Output = output;
            this.Input = input;
            this.Height = h;
            this.Width = w;
        }
    }

}
