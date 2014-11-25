using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using ManagedCuda.BasicTypes;

namespace CudaDnn
{
    public static class CudnnNativeMethods
    {
        internal const string ApiDllName = "cudnn64_65";


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreate(ref CudnnHandle handle);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroy(CudnnHandle handle);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetStream(CudnnHandle handle, CUstream streamId);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnGetStream(CudnnHandle handle, ref CUstream streamId);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreateTensor4dDescriptor(ref CudnnTensorDescriptor tensorDesc);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetTensor4dDescriptor(CudnnTensorDescriptor tensorDesc,
                                                                    CudnnTensorFormat format,
                                                                    CudnnType dataType,    // image data type
                                                                    int n,                 // number of inputs (batch size)
                                                                    int c,                 // number of input feature maps
                                                                    int h,                 // height of input section
                                                                    int w);                // width of input section

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetTensor4dDescriptorEx(CudnnTensorDescriptor tensorDesc,
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
        public static extern CudnnStatus cudnnGetTensor4dDescriptorEx(CudnnTensorDescriptor tensorDesc,
                                                                      ref CudnnType dataType,    // image data type
                                                                      ref int n,                 // number of inputs (batch size)
                                                                      ref int c,                 // number of input feature maps
                                                                      ref int h,                 // height of input section
                                                                      ref int w,
                                                                      ref int nStride,
                                                                      ref int cStride,
                                                                      ref int hStride,
                                                                      ref int wStride);



        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroyTensor4dDescriptor(CudnnTensorDescriptor tensorDesc);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnTransformTensor4d(CudnnHandle handle,
                                                                CudnnTensorDescriptor srcDescriptor,
                                                                [In] IntPtr srcData,
                                                                CudnnTensorDescriptor destDescriptor,
                                                                [In] IntPtr destData);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnAddTensor4d(CudnnHandle handle,
                                                          CudnnAdditionMode mode,
                                                          CudnnTensorDescriptor biasDescriptor,
                                                          [In] IntPtr biasData,
                                                          CudnnTensorDescriptor srcDestDescriptor,
                                                          [In, Out] IntPtr srcDestData);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetTensor4d(CudnnHandle handle,                                                 
                                                          CudnnTensorDescriptor tensorDescriptor,
                                                          [In, Out] IntPtr tensorData,
                                                          [In] IntPtr value);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreateFilterDescriptor(ref CudnnFilterDescriptor filterDescriptor);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroyFilterDescriptor(CudnnFilterDescriptor filterDescriptor);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetFilterDescriptor(CudnnFilterDescriptor filterDescriptor,
                                                                  CudnnType dataType,    // image data type
                                                                  int k,                 // number of output feature maps
                                                                  int c,                 // number of input feature maps
                                                                  int h,                 // height of each input filter
                                                                  int w                 // width of  each input fitler
                                                                 );

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnGetFilterDescriptor(CudnnFilterDescriptor filterDescriptor,
                                                                  ref CudnnType dataType,    // image data type
                                                                  ref int k,                 // number of output feature maps
                                                                  ref int c,                 // number of input feature maps
                                                                  ref int h,                 // height of each input filter
                                                                  ref int w                 // width of  each input fitler
                                                                 );


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreateConvolutionDescriptor(ref CudnnConvolutionDescriptor convolutionDescriptor);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroyConvolutionDescriptor(CudnnConvolutionDescriptor convolutionDescriptor);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetConvolutionDescriptor(
                                            CudnnConvolutionDescriptor convolutionDescriptor,
                                            CudnnTensorDescriptor inputTensorDescriptor,
                                            CudnnFilterDescriptor filterDescriptor,
                                            int paddingHeight,       // zero-padding height
                                            int paddingWidth,        // zero-padding width
                                            int verticalStride,      // vertical filter stride
                                            int horizontalStride,    // horizontal filter stride
                                            int upscaleVertical,     // upscale the input in x-direction
                                            int upscaleHorizontal,   // upscale the input in y-direction
                                            CudnnConvolutionMode mode);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetConvolutionDescriptorEx(
                                            CudnnConvolutionDescriptor convolutionDescriptor,
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
                                            CudnnConvolutionDescriptor convolutionDescriptor,
                                            ref CudnnConvolutionPath path,
                                            ref int n,                 
                                            ref int c,                
                                            ref int h,                
                                            ref int w);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnConvolutionForward(
                                            CudnnHandle handle,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnFilterDescriptor filterDescriptor,
                                            [In] IntPtr filterData,
                                            CudnnConvolutionDescriptor convolutionDescriptor,
                                            CudnnTensorDescriptor destDescriptor,
                                            [In, Out] IntPtr destData,
                                            CudnnAccumulateResult accumulate);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnConvolutionBackwardBias(
                                            CudnnHandle handle,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnTensorDescriptor destDescriptor,
                                            [In, Out] IntPtr destData,
                                            CudnnAccumulateResult accumulate);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnConvolutionBackwardFilter(
                                            CudnnHandle handle,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnTensorDescriptor diffDescriptor,
                                            [In] IntPtr diffData,
                                            CudnnConvolutionDescriptor convolutionDescriptor,
                                            CudnnFilterDescriptor gradientDescriptor,
                                            [In, Out] IntPtr gradientData,
                                            CudnnAccumulateResult accumulate);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnConvolutionBackwardData(
                                            CudnnHandle handle,
                                            CudnnFilterDescriptor filterDescriptor,
                                            [In] IntPtr filterData,
                                            CudnnTensorDescriptor diffDescriptor,
                                            [In] IntPtr diffData,
                                            CudnnConvolutionDescriptor convolutionDescriptor,
                                            CudnnTensorDescriptor gradientDescriptor,
                                            [In, Out] IntPtr gradientData,
                                            CudnnAccumulateResult accumulate);



        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSoftmaxForward(
                                            CudnnHandle handle,
                                            CudnnSoftmaxAlgorithm algorithm,
                                            CudnnSoftmaxMode mode,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnTensorDescriptor destDescriptor,
                                            [In, Out] IntPtr destData);




        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSoftmaxBackward(
                                            CudnnHandle handle,
                                            CudnnSoftmaxAlgorithm algorithm,
                                            CudnnSoftmaxMode mode,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnTensorDescriptor srcDiffDescriptor,
                                            [In] IntPtr srcDiffData,
                                            CudnnTensorDescriptor destDiffDescriptor,
                                            [In, Out] IntPtr destDiffData);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnCreatePoolingDescriptor(ref CudnnPoolingDescriptor poolingDescriptor);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnDestroyPoolingDescriptor(CudnnPoolingDescriptor poolingDescriptor);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnSetPoolingDescriptor(
                                            CudnnPoolingDescriptor poolingDescriptor,
                                            CudnnPoolingMode mode,
                                            int windowHeight,
                                            int windowWidth,
                                            int verticalStride,
                                            int horizontalStride);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnGetPoolingDescriptor(
                                            CudnnPoolingDescriptor poolingDescriptor,
                                            ref CudnnPoolingMode mode,
                                            ref int windowHeight,
                                            ref int windowWidth,
                                            ref int verticalStride,
                                            ref int horizontalStride);


        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnPoolingForward(
                                            CudnnHandle handle,
                                            CudnnPoolingDescriptor poolingDescriptor,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnTensorDescriptor destDescriptor,
                                            [In, Out] IntPtr destData);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnPoolingBackward(
                                            CudnnHandle handle,
                                            CudnnPoolingDescriptor poolingDescriptor,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnTensorDescriptor srcDiffDescriptor,
                                            [In] IntPtr srcDiffData,
                                            CudnnTensorDescriptor destDescriptor,
                                            [In] IntPtr destData,
                                            CudnnTensorDescriptor destDiffDescriptor,
                                            [In, Out] IntPtr destDiffData);




        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnActivationForward(
                                            CudnnHandle handle,
                                            CudnnActivationMode mode,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnTensorDescriptor destDescriptor,
                                            [In, Out] IntPtr destData);

        [DllImport(ApiDllName)]
        public static extern CudnnStatus cudnnActivationBackward(
                                            CudnnHandle handle,
                                            CudnnActivationMode mode,
                                            CudnnTensorDescriptor srcDescriptor,
                                            [In] IntPtr srcData,
                                            CudnnTensorDescriptor srcDiffDescriptor,
                                            [In] IntPtr srcDiffData,
                                            CudnnTensorDescriptor destDescriptor,
                                            [In] IntPtr destData,
                                            CudnnTensorDescriptor destDiffDescriptor,
                                            [In, Out] IntPtr destDiffData);
    }
}
