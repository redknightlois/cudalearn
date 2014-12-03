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
    public sealed class CudnnContext : CriticalFinalizerObject, IDisposable
    {
        #region Lifecycle 

        private CudnnHandle handle;

        protected CudnnContext( CudnnHandle handle )
        {
            if (handle.Pointer == IntPtr.Zero)
                throw new ArgumentException("handle");

            Contract.Ensures(this.handle.Pointer != IntPtr.Zero);
            Contract.EndContractBlock();

            this.handle = handle;
        }

        public static CudnnContext Create()
        {         
            CudnnHandle handle = default(CudnnHandle);            
            Invoke(() => CudnnNativeMethods.cudnnCreate(ref handle));
            return new CudnnContext(handle);
        }

        ~CudnnContext()
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

        private void DisposeManaged()
        {
        }

        private void DisposeNative()
        {
            try
            {
                if (this.handle.Pointer != IntPtr.Zero)
                {
                    Invoke(() => CudnnNativeMethods.cudnnDestroy(handle));
                }                    
            }
            finally
            {
                this.handle.Pointer = IntPtr.Zero;     
            }  
        }

        #endregion


        public static CudnnTensorDescriptor CreateTensor()
        {
            CudnnTensorDescriptorHandle handle = default(CudnnTensorDescriptorHandle);
            Invoke(() => CudnnNativeMethods.cudnnCreateTensor4dDescriptor(ref handle));
            return new CudnnTensorDescriptor(handle);
        }

        public static CudnnTensorDescriptor CreateTensor(CudnnTensorDescriptorParameters parameters)
        {
            var tensor = CreateTensor();
            tensor.SetParameters(parameters);
            return tensor;
        }

        public static CudnnFilterDescriptor CreateFilter()
        {
            CudnnFilterDescriptorHandle handle = default(CudnnFilterDescriptorHandle);
            Invoke(() => CudnnNativeMethods.cudnnCreateFilterDescriptor(ref handle));
            return new CudnnFilterDescriptor(handle);
        }

        public static CudnnFilterDescriptor CreateFilter(CudnnFilterDescriptorParameters parameters)
        {
            var filter = CreateFilter();
            filter.SetParameters(parameters);
            return filter;
        }

        public static CudnnPoolingDescriptor CreatePooling()
        {
            CudnnPoolingDescriptorHandle handle = default(CudnnPoolingDescriptorHandle);
            Invoke(() => CudnnNativeMethods.cudnnCreatePoolingDescriptor(ref handle));
            return new CudnnPoolingDescriptor(handle);
        }

        public static CudnnPoolingDescriptor CreatePooling(CudnnPoolingDescriptorParameters parameters)
        {
            var pooling = CreatePooling();
            pooling.SetParameters(parameters);
            return pooling;
        }

        public static CudnnConvolutionDescriptor CreateConvolution()
        {
            CudnnConvolutionDescriptorHandle handle = default(CudnnConvolutionDescriptorHandle);
            Invoke(() => CudnnNativeMethods.cudnnCreateConvolutionDescriptor(ref handle));
            return new CudnnConvolutionDescriptor(handle);
        }

        public static CudnnConvolutionDescriptor CreateConvolution(CudnnConvolutionDescriptorParameters parameters)
        {
            var Convolution = CreateConvolution();
            Convolution.SetParameters(parameters);
            return Convolution;
        }

        public static CudnnConvolutionDescriptor CreateConvolution(CudnnConvolutionDescriptorParametersEx parameters)
        {
            var Convolution = CreateConvolution();
            Convolution.SetParameters(parameters);
            return Convolution;
        }


        internal static void Invoke ( Func<CudnnStatus> action )
        {
            var result = action();
            switch (result)
            {
                case CudnnStatus.Success: return;
                case CudnnStatus.NotInitialized: throw new InvalidOperationException();
                case CudnnStatus.BadParameter: throw new ArgumentException("An incorrect value or parameter was passed to the function.");
                case CudnnStatus.InvalidValue: throw new ArgumentException("An incorrect value or parameter was passed to the function."); 
                case CudnnStatus.NotSupported: throw new NotSupportedException("The functionality requested is not presently supported by cuDNN.");
                case CudnnStatus.ArchitectureMismatch: throw new NotSupportedException("The function requires a feature absent from the current GPU device. Note that cuDNN only supports devices with compute capabilities greater than or equal to 3.0.");
                case CudnnStatus.AllocationFailed: throw new OutOfMemoryException("Resource allocation failed inside the cuDNN library. This is usually caused by an internal cudaMalloc() failure. Deallocate previously allocated memory as much as possible.");
                case CudnnStatus.ExecutionFailed: throw new CudnnException(result, "The GPU program failed to execute. This is usually caused by a failure to launch some cuDNN kernel on the GPU, which can occur for multiple reasons");
                case CudnnStatus.InternalError: throw new CudnnException(result, "An internal cuDNN operation failed. Probably a cuDNN bug.");
                case CudnnStatus.LicenseError: throw new CudnnException(result, "The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.");
                case CudnnStatus.MappingError: throw new CudnnException(result, "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.");                                
            }
        }
    }

}
