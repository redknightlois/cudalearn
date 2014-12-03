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
    public sealed class CudnnConvolutionDescriptor : CriticalFinalizerObject, IDisposable
    {
        internal CudnnConvolutionDescriptorHandle Handle;
        private bool initialized = false;

        internal CudnnConvolutionDescriptor(CudnnConvolutionDescriptorHandle handle)
        {
            this.Handle = handle;
        }

        ~CudnnConvolutionDescriptor()
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

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnDestroyConvolutionDescriptor(this.Handle));
            this.Handle.Pointer = IntPtr.Zero;
        }

        private void ThrowIfNotInitialized()
        {
            if (!IsInitialized)
                throw new InvalidOperationException("Not initialized.");
        }

        public bool IsInitialized
        {
            get { return this.Handle.Pointer != IntPtr.Zero && initialized; }
        }

        public void SetParameters(CudnnConvolutionDescriptorParameters param)
        {
            if (param == null)
                throw new ArgumentNullException("param");

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnSetConvolutionDescriptor(
                                                    this.Handle, param.Tensor.Handle, param.Filter.Handle,
                                                    param.HeightPadding, param.WidthPadding,
                                                    param.HeightStride, param.WidthStride,
                                                    param.HeightUpscale, param.WidthUpscale,
                                                    param.Mode));

            initialized = true;
        }


        public void SetParameters(CudnnConvolutionDescriptorParametersEx param)
        {
            if (param == null)
                throw new ArgumentNullException("param");

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnSetConvolutionDescriptorEx(
                                                    this.Handle, 
                                                    param.Num, param.Channels, param.Height, param.Width,
                                                    param.Kernel, param.FilterHeight, param.FilterWidth,
                                                    param.HeightPadding, param.WidthPadding,
                                                    param.HeightStride, param.WidthStride,
                                                    param.HeightUpscale, param.WidthUpscale,
                                                    param.Mode));

            initialized = true;
        }

        public CudnnConvolutionTensorDim GetOutputTensor(CudnnConvolutionPath path)
        {
            ThrowIfNotInitialized();

            int n = 0, c = 0, h = 0, w = 0;
            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetOutputTensor4dDim(this.Handle, path, out n, out c, out h, out w));

            return new CudnnConvolutionTensorDim(path, n, c, h, w);
        }
    }


    public class CudnnConvolutionDescriptorParameters
    {
        public readonly CudnnConvolutionMode Mode;

        public readonly CudnnTensorDescriptor Tensor;
        public readonly CudnnFilterDescriptor Filter;
        public readonly int HeightPadding;
        public readonly int WidthPadding;

        public readonly int HeightStride;
        public readonly int WidthStride;

        public readonly int HeightUpscale;
        public readonly int WidthUpscale;

        public CudnnConvolutionDescriptorParameters(CudnnConvolutionMode mode, CudnnTensorDescriptor tensor, CudnnFilterDescriptor filter, 
                                                    int paddingHeight, int paddingWidth,
                                                    int verticalStride, int horizontalStride,
                                                    int upscaleVertical = 1, int upscaleHorizontal = 1)
        {
            if (tensor == null)
                throw new ArgumentNullException("tensor");

            if (filter == null)
                throw new ArgumentNullException("tensor");

            if (upscaleVertical != 1 || upscaleHorizontal != 1)
                throw new NotSupportedException("The parameter upscaleVertical or upscaleHorizontal is not 1");

            if (verticalStride < 1 || horizontalStride < 1)
                throw new ArgumentException("One of the parameters verticalStride or horizontalStride is negative.");

            if (tensor.Parameters.Type != filter.Parameters.Type)
                throw new ArgumentException("The Type of the tensor and filter descriptors differ or have invalid values.");

            if (filter.Parameters.Input != tensor.Parameters.Channels)
                throw new ArgumentException("The number of feature maps of the tensor descriptor and the number of input feature maps of the filter differ.");

            Contract.EndContractBlock();

            this.Mode = mode;
            this.Tensor = tensor;
            this.Filter = filter;

            this.HeightPadding = paddingHeight;
            this.WidthPadding = paddingWidth;
            this.HeightStride = verticalStride;
            this.WidthStride = horizontalStride;
            this.HeightUpscale = upscaleVertical;
            this.WidthUpscale = upscaleHorizontal;
        }
    }

    public class CudnnConvolutionDescriptorParametersEx
    {
        public readonly CudnnConvolutionMode Mode;

        public readonly int Num;
        public readonly int Channels;
        public readonly int Height;
        public readonly int Width;

        public readonly int Kernel;
        public readonly int FilterHeight;
        public readonly int FilterWidth;

        public readonly int HeightPadding;
        public readonly int WidthPadding;

        public readonly int HeightStride;
        public readonly int WidthStride;

        public readonly int HeightUpscale;
        public readonly int WidthUpscale;

        public CudnnConvolutionDescriptorParametersEx(CudnnConvolutionMode mode,
                                    int n, int c, int h, int w,
                                    int k, int r, int s,
                                    int paddingHeight, int paddingWidth,
                                    int verticalStride, int horizontalStride,
                                    int upscaleVertical = 1, int upscaleHorizontal = 1)
        {
            if (upscaleVertical != 1 || upscaleHorizontal != 1)
                throw new NotSupportedException("The parameter upscaleVertical or upscaleHorizontal is not 1");

            if (verticalStride < 1 || horizontalStride < 1)
                throw new ArgumentException("One of the parameters verticalStride or horizontalStride is negative.");

            Contract.EndContractBlock();

            this.Mode = mode;

            this.Num = n;
            this.Channels = c;
            this.Height = h;
            this.Width = w;

            this.Kernel = k;
            this.FilterHeight = r;
            this.FilterWidth = s;

            this.HeightPadding = paddingHeight;
            this.WidthPadding = paddingWidth;
            this.HeightStride = verticalStride;
            this.WidthStride = horizontalStride;
            this.HeightUpscale = upscaleVertical;
            this.WidthUpscale = upscaleHorizontal;        
        }
    }
}
