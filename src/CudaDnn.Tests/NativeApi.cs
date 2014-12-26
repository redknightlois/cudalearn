using CudaDnn.Impl;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaDnn.Tests
{
    public class NativeApiTests : CudaDnnTestBase
    {

        [Fact]
        public void Cudnn_NativeApi_HandleLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(out handle));
            Assert.NotNull(handle);
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }

        [Fact]
        public void Cudnn_NativeApi_FilterLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(out handle));

            CudnnFilterDescriptorHandle filterDescriptor = default(CudnnFilterDescriptorHandle);
            Success(() => CudnnNativeMethods.cudnnCreateFilterDescriptor(out filterDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroyFilterDescriptor(filterDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }

        [Fact]
        public void Cudnn_NativeApi_ConvolutionLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(out handle));

            CudnnConvolutionDescriptorHandle convolutionDescriptor = default(CudnnConvolutionDescriptorHandle);
            Success(() => CudnnNativeMethods.cudnnCreateConvolutionDescriptor(out convolutionDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroyConvolutionDescriptor(convolutionDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }

        [Fact]
        public void Cudnn_NativeApi_PoolingLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(out handle));

            CudnnPoolingDescriptorHandle poolingDescriptor = default(CudnnPoolingDescriptorHandle);
            Success(() => CudnnNativeMethods.cudnnCreatePoolingDescriptor(out poolingDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroyPoolingDescriptor(poolingDescriptor));
            Success(() => CudnnNativeMethods.cudnnDestroy(handle));
        }

        [Fact]
        public void Cudnn_NativeApi_TensorLifecycle()
        {
            CudnnHandle handle = default(CudnnHandle);
            Success(() => CudnnNativeMethods.cudnnCreate(out handle));

            CudnnTensorDescriptorHandle tensorDescriptor = default(CudnnTensorDescriptorHandle);
            Success(() => CudnnNativeMethods.cudnnCreateTensor4dDescriptor(out tensorDescriptor));
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
