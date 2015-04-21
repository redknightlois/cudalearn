using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using ManagedCuda.BasicTypes;

namespace CudaDnn.Impl
{
    internal static class CudnnNativeMethods
    {
        internal const string ApiDllName = "cudnn64_65";


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreate(out CudnnHandle handle);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroy(CudnnHandle handle);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetStream(CudnnHandle handle, CUstream streamId);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnGetStream(CudnnHandle handle, out CUstream streamId);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreateTensor4dDescriptor(out CudnnTensorDescriptorHandle tensorDesc);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetTensor4dDescriptor(CudnnTensorDescriptorHandle tensorDesc,
                                                                    CudnnTensorFormat format,
                                                                    CudnnType dataType,    // image data type
                                                                    int n,                 // number of inputs (batch size)
                                                                    int c,                 // number of input feature maps
                                                                    int h,                 // height of input section
                                                                    int w);                // width of input section

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetTensor4dDescriptorEx(CudnnTensorDescriptorHandle tensorDesc,
                                                                      CudnnType dataType,    // image data type
                                                                      int n,                 // number of inputs (batch size)
                                                                      int c,                 // number of input feature maps
                                                                      int h,                 // height of input section
                                                                      int w,
                                                                      int nStride,
                                                                      int cStride,
                                                                      int hStride,
                                                                      int wStride);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnGetTensor4dDescriptor(CudnnTensorDescriptorHandle tensorDesc,
                                                                      out CudnnType dataType,    // image data type
                                                                      out int n,                 // number of inputs (batch size)
                                                                      out int c,                 // number of input feature maps
                                                                      out int h,                 // height of input section
                                                                      out int w,
                                                                      out int nStride,
                                                                      out int cStride,
                                                                      out int hStride,
                                                                      out int wStride);



        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroyTensor4dDescriptor(CudnnTensorDescriptorHandle tensorDesc);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnTransformTensor4d(CudnnHandle handle,
                                                                CudnnTensorDescriptorHandle srcDescriptor,
                                                                [In] CUdeviceptr srcData,
                                                                CudnnTensorDescriptorHandle destDescriptor,
                                                                [In] CUdeviceptr destData);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnAddTensor4d(CudnnHandle handle,
                                                          CudnnAdditionMode mode,
                                                          CudnnTensorDescriptorHandle biasDescriptor,
                                                          [In] CUdeviceptr biasData,
                                                          CudnnTensorDescriptorHandle srcDestDescriptor,
                                                          [In, Out] CUdeviceptr srcDestData);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetTensor4d(CudnnHandle handle,                                                 
                                                          CudnnTensorDescriptorHandle tensorDescriptor,
                                                          [In, Out] CUdeviceptr tensorData,
                                                          [In] CUdeviceptr value);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreateFilterDescriptor(out CudnnFilterDescriptorHandle filterDescriptor);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroyFilterDescriptor(CudnnFilterDescriptorHandle filterDescriptor);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetFilterDescriptor(CudnnFilterDescriptorHandle filterDescriptor,
                                                                  CudnnType dataType,    // image data type
                                                                  int k,                 // number of output feature maps
                                                                  int c,                 // number of input feature maps
                                                                  int h,                 // height of each input filter
                                                                  int w                 // width of  each input filter
                                                                 );

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnGetFilterDescriptor(CudnnFilterDescriptorHandle filterDescriptor,
                                                                  out CudnnType dataType,    // image data type
                                                                  out int k,                 // number of output feature maps
                                                                  out int c,                 // number of input feature maps
                                                                  out int h,                 // height of each input filter
                                                                  out int w                 // width of  each input filter
                                                                 );


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreateConvolutionDescriptor(out CudnnConvolutionDescriptorHandle convolutionDescriptor);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroyConvolutionDescriptor(CudnnConvolutionDescriptorHandle convolutionDescriptor);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetConvolutionDescriptor(
                                            CudnnConvolutionDescriptorHandle convolutionDescriptor,
                                            CudnnTensorDescriptorHandle inputTensorDescriptor,
                                            CudnnFilterDescriptorHandle filterDescriptor,
                                            int paddingHeight,       // zero-padding height
                                            int paddingWidth,        // zero-padding width
                                            int verticalStride,      // vertical filter stride
                                            int horizontalStride,    // horizontal filter stride
                                            int upscaleVertical,     // upscale the input in x-direction
                                            int upscaleHorizontal,   // upscale the input in y-direction
                                            CudnnConvolutionMode mode);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetConvolutionDescriptorEx(
                                            CudnnConvolutionDescriptorHandle convolutionDescriptor,
                                            int n,
                                            int c,
                                            int h,
                                            int w,
                                            int k,
                                            int r,
                                            int s,
                                            int paddingHeight,       // zero-padding height
                                            int paddingWidth,        // zero-padding width
                                            int verticalStride,      // vertical filter stride
                                            int horizontalStride,    // horizontal filter stride
                                            int upscaleVertical,     // upscale the input in x-direction
                                            int upscaleHorizontal,   // upscale the input in y-direction
                                            CudnnConvolutionMode mode);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnGetOutputTensor4dDim(
                                            CudnnConvolutionDescriptorHandle convolutionDescriptor,
                                            CudnnConvolutionPath path,
                                            out int n,
                                            out int c,
                                            out int h,
                                            out int w);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnConvolutionForward(
                                            CudnnHandle handle,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnFilterDescriptorHandle filterDescriptor,
                                            [In] CUdeviceptr filterData,
                                            CudnnConvolutionDescriptorHandle convolutionDescriptor,
                                            CudnnTensorDescriptorHandle destDescriptor,
                                            [In, Out] CUdeviceptr destData,
                                            CudnnAccumulateResult accumulate);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnConvolutionBackwardBias(
                                            CudnnHandle handle,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnTensorDescriptorHandle destDescriptor,
                                            [In, Out] CUdeviceptr destData,
                                            CudnnAccumulateResult accumulate);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnConvolutionBackwardFilter(
                                            CudnnHandle handle,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnTensorDescriptorHandle diffDescriptor,
                                            [In] CUdeviceptr diffData,
                                            CudnnConvolutionDescriptorHandle convolutionDescriptor,
                                            CudnnFilterDescriptorHandle gradientDescriptor,
                                            [In, Out] CUdeviceptr gradientData,
                                            CudnnAccumulateResult accumulate);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnConvolutionBackwardData(
                                            CudnnHandle handle,
                                            CudnnFilterDescriptorHandle filterDescriptor,
                                            [In] CUdeviceptr filterData,
                                            CudnnTensorDescriptorHandle diffDescriptor,
                                            [In] CUdeviceptr diffData,
                                            CudnnConvolutionDescriptorHandle convolutionDescriptor,
                                            CudnnTensorDescriptorHandle gradientDescriptor,
                                            [In, Out] CUdeviceptr gradientData,
                                            CudnnAccumulateResult accumulate);



        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSoftmaxForward(
                                            CudnnHandle handle,
                                            CudnnSoftmaxAlgorithm algorithm,
                                            CudnnSoftmaxMode mode,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnTensorDescriptorHandle destDescriptor,
                                            [In, Out] CUdeviceptr destData);




        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSoftmaxBackward(
                                            CudnnHandle handle,
                                            CudnnSoftmaxAlgorithm algorithm,
                                            CudnnSoftmaxMode mode,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnTensorDescriptorHandle srcDiffDescriptor,
                                            [In] CUdeviceptr srcDiffData,
                                            CudnnTensorDescriptorHandle destDiffDescriptor,
                                            [In, Out] CUdeviceptr destDiffData);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreatePoolingDescriptor(out CudnnPoolingDescriptorHandle poolingDescriptor);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroyPoolingDescriptor(CudnnPoolingDescriptorHandle poolingDescriptor);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetPoolingDescriptor(
                                            CudnnPoolingDescriptorHandle poolingDescriptor,
                                            CudnnPoolingMode mode,
                                            int windowHeight,
                                            int windowWidth,
                                            int verticalStride,
                                            int horizontalStride);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnGetPoolingDescriptor(
                                            CudnnPoolingDescriptorHandle poolingDescriptor,
                                            out CudnnPoolingMode mode,
                                            out int windowHeight,
                                            out int windowWidth,
                                            out int verticalStride,
                                            out int horizontalStride);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnPoolingForward(
                                            CudnnHandle handle,
                                            CudnnPoolingDescriptorHandle poolingDescriptor,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnTensorDescriptorHandle destDescriptor,
                                            [In, Out] CUdeviceptr destData);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnPoolingBackward(
                                            CudnnHandle handle,
                                            CudnnPoolingDescriptorHandle poolingDescriptor,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnTensorDescriptorHandle srcDiffDescriptor,
                                            [In] CUdeviceptr srcDiffData,
                                            CudnnTensorDescriptorHandle destDescriptor,
                                            [In] CUdeviceptr destData,
                                            CudnnTensorDescriptorHandle destDiffDescriptor,
                                            [In, Out] CUdeviceptr destDiffData);




        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnActivationForward(
                                            CudnnHandle handle,
                                            CudnnActivationMode mode,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnTensorDescriptorHandle destDescriptor,
                                            [In, Out] CUdeviceptr destData);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnActivationBackward(
                                            CudnnHandle handle,
                                            CudnnActivationMode mode,
                                            CudnnTensorDescriptorHandle srcDescriptor,
                                            [In] CUdeviceptr srcData,
                                            CudnnTensorDescriptorHandle srcDiffDescriptor,
                                            [In] CUdeviceptr srcDiffData,
                                            CudnnTensorDescriptorHandle destDescriptor,
                                            [In] CUdeviceptr destData,
                                            CudnnTensorDescriptorHandle destDiffDescriptor,
                                            [In, Out] CUdeviceptr destDiffData);
    }
}
