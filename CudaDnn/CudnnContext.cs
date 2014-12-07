using CudaDnn.Impl;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using System.Text;
using System.Threading.Tasks;

namespace CudaDnn
{
    public sealed partial class CudnnContext : CriticalFinalizerObject, IDisposable
    {
        #region Lifecycle 

        private CudnnHandle handle;

        static CudnnContext ()
        {
            DefaultType = CudnnType.Double;
            DefaultTensorFormat = CudnnTensorFormat.MajorRow;
        }

        private CudnnContext( CudnnHandle handle )
        {
            if (handle.Pointer == IntPtr.Zero)
                throw new ArgumentException("handle");

            Contract.EndContractBlock();

            this.handle = handle;
        }

        public static CudnnContext Create()
        {         
            CudnnHandle handle = default(CudnnHandle);            
            Invoke(() => CudnnNativeMethods.cudnnCreate(out handle));
            Contract.Assume(handle.Pointer != IntPtr.Zero);

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

        public static CudnnType DefaultType { get; set; }

        public static CudnnTensorFormat DefaultTensorFormat { get; set; }

        private void ThrowIfNotInitialized()
        {
            if (!IsInitialized)
                throw new InvalidOperationException("Not initialized.");
        }

        public bool IsInitialized
        {
            get { return this.handle.Pointer != IntPtr.Zero; }
        }


        public static CudnnTensorDescriptor CreateTensor()
        {
            CudnnTensorDescriptorHandle handle = default(CudnnTensorDescriptorHandle);
            Invoke(() => CudnnNativeMethods.cudnnCreateTensor4dDescriptor(out handle));
            Contract.Assume(handle.Pointer != IntPtr.Zero);

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
            Invoke(() => CudnnNativeMethods.cudnnCreateFilterDescriptor(out handle));
            Contract.Assume(handle.Pointer != IntPtr.Zero);

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
            Invoke(() => CudnnNativeMethods.cudnnCreatePoolingDescriptor(out handle));
            Contract.Assume(handle.Pointer != IntPtr.Zero);

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
            Invoke(() => CudnnNativeMethods.cudnnCreateConvolutionDescriptor(out handle));
            Contract.Assume(handle.Pointer != IntPtr.Zero);

            return new CudnnConvolutionDescriptor(handle);
        }

        public static CudnnConvolutionDescriptor CreateConvolution(CudnnConvolutionDescriptorParameters parameters)
        {
            var convolution = CreateConvolution();
            convolution.SetParameters(parameters);
            return convolution;
        }

        public static CudnnConvolutionDescriptor CreateConvolution(CudnnConvolutionDescriptorParametersEx parameters)
        {
            var convolution = CreateConvolution();
            convolution.SetParameters(parameters);
            return convolution;
        }


        internal static void Invoke ( Func<CudnnStatus> action )
        {
            Contract.Requires(action != null);

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

        private static void CheckIfCompatible(CudnnType type, params ITypeAware[] parameters)

        {
            Contract.Requires(parameters != null);
            Contract.ForAll<ITypeAware>(parameters, x => x != null && x.Type == type);
            
            foreach (var param in parameters)
            {
                if (param == null)
                    throw new ArgumentNullException("Parameter is null");

                if (param.Type != type)
                    throw new ArgumentException(string.Format("One of the descriptors with type {0} is not CudnnType.{1}", param.GetType().Name, type.ToString()));
            }
        }

    }

}
