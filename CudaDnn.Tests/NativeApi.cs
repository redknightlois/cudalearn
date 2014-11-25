using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaDnn.Tests
{
    public class NativeApi
    {
        public NativeApi ()
        {
            Assert.True(Environment.Is64BitProcess, "Tests are being run as 32bits processes. CuDNN is not supported on 32bits. Change the setting in Test->Test Settings->Default Processor Architecture->x64." );
        }


        [Fact]
        public void Cudnn_NativeApi_HandleLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(ref handle));
            Assert.NotNull(handle);
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }

        [Fact]
        public void Cudnn_NativeApi_FilterLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(ref handle));

            CudnnFilterDescriptor filterDescriptor = default(CudnnFilterDescriptor);
            Success(() => CudnnNativeMethods.cudnnCreateFilterDescriptor(ref filterDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroyFilterDescriptor(filterDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }

        [Fact]
        public void Cudnn_NativeApi_ConvolutionLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(ref handle));

            CudnnConvolutionDescriptor convolutionDescriptor = default(CudnnConvolutionDescriptor);
            Success(() => CudnnNativeMethods.cudnnCreateConvolutionDescriptor(ref convolutionDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroyConvolutionDescriptor(convolutionDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }

        [Fact]
        public void Cudnn_NativeApi_PoolingLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(ref handle));

            CudnnPoolingDescriptor poolingDescriptor = default(CudnnPoolingDescriptor);
            Success(() => CudnnNativeMethods.cudnnCreatePoolingDescriptor(ref poolingDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroyPoolingDescriptor(poolingDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }

        [Fact]
        public void Cudnn_NativeApi_TensorLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(ref handle));

            CudnnTensorDescriptor tensorDescriptor = default(CudnnTensorDescriptor);
            Success(() => CudnnNativeMethods.cudnnCreateTensor4dDescriptor(ref tensorDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroyTensor4dDescriptor(tensorDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }


        private void Success(Func<CudnnStatus> action)
        {
            var result = action();
            Assert.Equal(CudnnStatus.Success, result);
        }

        private void Fail(Func<CudnnStatus> action)
        {
            var result = action();
            Assert.NotEqual(CudnnStatus.Success, result);
        }
    }
}
