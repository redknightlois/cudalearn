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
    public sealed class CudnnTensorDescriptor : CriticalFinalizerObject, IDisposable
    {
        internal CudnnTensorDescriptorHandle Handle;

        internal CudnnTensorDescriptor(CudnnTensorDescriptorHandle handle)
        {
            if (handle.Pointer == IntPtr.Zero)
                throw new ArgumentException("The handle pointer is null.", "handle");

            Contract.EndContractBlock();

            this.Handle = handle;
        }

        ~CudnnTensorDescriptor()
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
            Contract.EndContractBlock();

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnDestroyTensor4dDescriptor(this.Handle));
            this.Handle.Pointer = IntPtr.Zero;
        }


        private CudnnTensorDescriptorParameters descriptorParams;

        public CudnnTensorDescriptorParameters Parameters
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

        public void SetParameters(CudnnTensorDescriptorParameters param)
        {
            if (param == null)
                throw new ArgumentNullException("param");

            CudnnContext.Invoke(() => CudnnNativeMethods.cudnnSetTensor4dDescriptorEx(
                                            this.Handle, param.Type,
                                            param.Num, param.Channels, param.Height, param.Width,
                                            param.NumStride, param.ChannelsStride, param.HeightStride, param.WidthStride));

            this.descriptorParams = param;
        }
    }

    public class CudnnTensorDescriptorParameters
    {
        public readonly CudnnType Type;

        public readonly int Num;
        public readonly int Channels;
        public readonly int Height;
        public readonly int Width;

        public readonly int NumStride;
        public readonly int ChannelsStride;
        public readonly int HeightStride;
        public readonly int WidthStride;

        public CudnnTensorDescriptorParameters(CudnnType type, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride)
        {
            if (n < 1 || c < 1 || h < 1 || w < 1)
                throw new ArgumentException("At least one of the parameters n, c, h, w was negative.");

            if (nStride < 1 || cStride < 1 || hStride < 1 || wStride < 1)
                throw new ArgumentException("At least one of the parameters nStride, cStride,h Stride, wStride is negative.");

            this.Type = type;

            this.Num = n;
            this.Channels = c;
            this.Height = h;
            this.Width = w;

            this.NumStride = nStride;
            this.ChannelsStride = cStride;
            this.HeightStride = hStride;
            this.WidthStride = wStride;
        }

        public CudnnTensorDescriptorParameters(CudnnType type, CudnnTensorFormat format, int n, int c, int h, int w)
        {
            if (n < 1 || c < 1 || h < 1 || w < 1)
                throw new ArgumentException("At least one of the parameters n, c, h, w was negative.");

            this.Type = type;

            this.Num = n;
            this.Channels = c;
            this.Height = h;
            this.Width = w;
            
            this.NumStride = h * w * c;

            if ( format == CudnnTensorFormat.MajorRow )
            {
                this.ChannelsStride = h * w;
                this.HeightStride = w;
                this.WidthStride = 1;
            }
            else
            {
                this.ChannelsStride = 1;
                this.HeightStride = w * c;
                this.WidthStride = c;
            }
        }
    }
}
