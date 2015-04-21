using CudaDnn.Impl;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaDnn.Tests
{
    public class CudnnDescriptorsTests : CudaDnnTestBase
    {

        [Fact]
        public void Cudnn_Descriptors_Constructors()
        {
            using (var tensor = CudnnContext.CreateTensor())
            using (var convolution = CudnnContext.CreateConvolution())
            using (var pooling = CudnnContext.CreatePooling())
            using (var filter = CudnnContext.CreateFilter())
            {
                Assert.False(tensor.IsInitialized);
                Assert.False(convolution.IsInitialized);
                Assert.False(pooling.IsInitialized);
                Assert.False(filter.IsInitialized);
            }
        }

        public static IEnumerable<object[]> TensorConfigurations
        {
            get
            {
                return new[]
                {    
                    new object[] { new CudnnTensorDescriptorParameters(CudnnType.Float, CudnnTensorFormat.MajorRow, 10, 5, 2, 4) },                                                   
                    new object[] { new CudnnTensorDescriptorParameters(CudnnType.Double, CudnnTensorFormat.Interleaved, 4, 5, 6, 2) },                                                   
                };
            }
        }


        [Theory, MemberData("TensorConfigurations")]
        public void Cudnn_Descriptors_ConstructTensorWithSetup(CudnnTensorDescriptorParameters param)
        {
            using (var tensor = CudnnContext.CreateTensor(param))
            {
                Assert.True(tensor.IsInitialized);

                CudnnType dataType = default(CudnnType);
                int n = 0, c = 0, h = 0, w = 0;
                int nStride = 0, cStride = 0, hStride = 0, wStride = 0;
                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetTensor4dDescriptor(tensor.Handle, out dataType, out n, out c, out h, out w, out nStride, out cStride, out hStride, out wStride));

                Assert.Equal(tensor.Parameters.Num, n);
                Assert.Equal(tensor.Parameters.Channels, c);
                Assert.Equal(tensor.Parameters.Height, h);
                Assert.Equal(tensor.Parameters.Width, w);

                Assert.Equal(tensor.Parameters.NumStride, nStride);
                Assert.Equal(tensor.Parameters.ChannelsStride, cStride);
                Assert.Equal(tensor.Parameters.HeightStride, hStride);
                Assert.Equal(tensor.Parameters.WidthStride, wStride);
            }
        }

        [Fact]
        public void Cudnn_Descriptors_ConstructTensorWithMajorRow()
        {
            var param = new CudnnTensorDescriptorParameters(CudnnType.Float, CudnnTensorFormat.MajorRow, 10, 5, 2, 4);
            using (var tensor = CudnnContext.CreateTensor(param))
            using (var tensor2 = CudnnContext.CreateTensor())
            {
                Assert.True(tensor.IsInitialized);
                Assert.False(tensor2.IsInitialized);

                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnSetTensor4dDescriptor(tensor2.Handle, CudnnTensorFormat.MajorRow, CudnnType.Float, 10, 5, 2, 4));


                CudnnType dataType = default(CudnnType);
                int n1 = 0, c1 = 0, h1 = 0, w1 = 0;
                int nStride1 = 0, cStride1 = 0, hStride1 = 0, wStride1 = 0;
                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetTensor4dDescriptor(tensor.Handle, out dataType, out n1, out c1, out h1, out w1, out nStride1, out cStride1, out hStride1, out wStride1));

                int n2 = 0, c2 = 0, h2 = 0, w2 = 0;
                int nStride2 = 0, cStride2 = 0, hStride2 = 0, wStride2 = 0;
                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetTensor4dDescriptor(tensor2.Handle, out dataType, out n2, out c2, out h2, out w2, out nStride2, out cStride2, out hStride2, out wStride2));

                Assert.Equal(n2, n1);
                Assert.Equal(c2, c1);
                Assert.Equal(h2, h1);
                Assert.Equal(w2, w1);

                Assert.Equal(nStride2, nStride1);
                Assert.Equal(cStride2, cStride1);
                Assert.Equal(hStride2, hStride1);
                Assert.Equal(wStride2, wStride1);
            }
        }

        [Fact]
        public void Cudnn_Descriptors_ConstructTensorWithInterleaved_Current()
        {
            using (var tensor = CudnnContext.CreateTensor())
            {
                Assert.False(tensor.IsInitialized);
                Assert.Throws<NotSupportedException>(
                    () => CudnnContext.Invoke(() => CudnnNativeMethods.cudnnSetTensor4dDescriptor(tensor.Handle, CudnnTensorFormat.Interleaved, CudnnType.Double, 10, 5, 2, 4)));
            }
        }

        [Fact(Skip = "Not Supported by CuDNN yet.")]
        public void Cudnn_Descriptors_ConstructTensorWithInterleaved_Support()
        {
            var param = new CudnnTensorDescriptorParameters(CudnnType.Double, CudnnTensorFormat.Interleaved, 10, 5, 2, 4);
            using (var tensor = CudnnContext.CreateTensor(param))
            using (var tensor2 = CudnnContext.CreateTensor())
            {
                Assert.True(tensor.IsInitialized);
                Assert.False(tensor2.IsInitialized);

                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnSetTensor4dDescriptor(tensor2.Handle, CudnnTensorFormat.Interleaved, CudnnType.Double, 10, 5, 2, 4));


                CudnnType dataType = default(CudnnType);
                int n1 = 0, c1 = 0, h1 = 0, w1 = 0;
                int nStride1 = 0, cStride1 = 0, hStride1 = 0, wStride1 = 0;
                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetTensor4dDescriptor(tensor.Handle, out dataType, out n1, out c1, out h1, out w1, out nStride1, out cStride1, out hStride1, out wStride1));

                int n2 = 0, c2 = 0, h2 = 0, w2 = 0;
                int nStride2 = 0, cStride2 = 0, hStride2 = 0, wStride2 = 0;
                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetTensor4dDescriptor(tensor2.Handle, out dataType, out n2, out c2, out h2, out w2, out nStride2, out cStride2, out hStride2, out wStride2));

                Assert.Equal(n2, n1);
                Assert.Equal(c2, c1);
                Assert.Equal(h2, h1);
                Assert.Equal(w2, w1);

                Assert.Equal(nStride2, nStride1);
                Assert.Equal(cStride2, cStride1);
                Assert.Equal(hStride2, hStride1);
                Assert.Equal(wStride2, wStride1);
            }
        }

        public static IEnumerable<CudnnConvolutionDescriptorParameters> ConvolutionConfigurations(CudnnTensorDescriptor tensor, CudnnFilterDescriptor filter)
        {
            yield return new CudnnConvolutionDescriptorParameters(CudnnConvolutionMode.Convolution, tensor, filter, 1, 1, 3, 3);
            yield return new CudnnConvolutionDescriptorParameters(CudnnConvolutionMode.CrossCorrelation, tensor, filter, 1, 1, 3, 3);            
        }

        public static IEnumerable<object[]> ConvolutionConfigurationsEx
        {
            get
            {
                return new[]
                {    
                    new object[] { new CudnnConvolutionDescriptorParametersEx(CudnnConvolutionMode.Convolution, 10, 3, 3, 4, 4, 2, 2, 1, 1, 1, 1) },                                                   
                    new object[] { new CudnnConvolutionDescriptorParametersEx(CudnnConvolutionMode.Convolution, 6, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1) },                                                   
                    new object[] { new CudnnConvolutionDescriptorParametersEx(CudnnConvolutionMode.CrossCorrelation, 10, 3, 3, 4, 4, 2, 2, 1, 1, 1, 1) },                                                   
                    new object[] { new CudnnConvolutionDescriptorParametersEx(CudnnConvolutionMode.CrossCorrelation, 6, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1) },                                                   
                };
            }
        }

        [Fact]
        public void Cudnn_Descriptors_ConstructConvolutionWithSetup()
        {
            using(var filter = CudnnContext.CreateFilter(new CudnnFilterDescriptorParameters(CudnnType.Float, 10, 10, 2, 2)))
            using (var tensor = CudnnContext.CreateTensor(new CudnnTensorDescriptorParameters(CudnnType.Float, CudnnTensorFormat.MajorRow, 10, 10, 100, 100)))
            {
                var param = new CudnnConvolutionDescriptorParameters(CudnnConvolutionMode.Convolution, tensor, filter, 1, 1, 3, 3, 1, 1);
                using (var convolution = CudnnContext.CreateConvolution(param))
                {
                    Assert.True(convolution.IsInitialized);

                    var dimensions = convolution.GetOutputTensor(CudnnConvolutionPath.Forward);
                    Assert.NotNull(dimensions);
                }            
            }
        }

        [Theory, MemberData("ConvolutionConfigurationsEx")]
        public void Cudnn_Descriptors_ConstructConvolutionWithSetupEx(CudnnConvolutionDescriptorParametersEx param)
        {
            using (var convolution = CudnnContext.CreateConvolution(param))
            {
                Assert.True(convolution.IsInitialized);

                var dimensions = convolution.GetOutputTensor(CudnnConvolutionPath.Forward);
                Assert.NotNull(dimensions);
            }
        }

        public static IEnumerable<object[]> PoolingConfigurations
        {
            get
            {
                return new[]
                {    
                    new object[] { new CudnnPoolingDescriptorParameters(CudnnPoolingMode.Average, 6, 5, 4, 3) },                                                   
                    new object[] { new CudnnPoolingDescriptorParameters(CudnnPoolingMode.Max, 3, 4, 5, 6) },                                                   
                };
            }
        }


        [Theory, MemberData("PoolingConfigurations")]
        public void Cudnn_Descriptors_ConstructPoolingWithSetup(CudnnPoolingDescriptorParameters param)
        {
            using (var pooling = CudnnContext.CreatePooling(param))
            {
                Assert.True(pooling.IsInitialized);

                CudnnPoolingMode mode = default(CudnnPoolingMode);
                int windowHeight = 0, windowWidth = 0;
                int verticalStride = 0, horizontalStride = 0;

                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetPoolingDescriptor(pooling.Handle, out mode, out windowHeight, out windowWidth, out verticalStride, out horizontalStride));

                Assert.Equal(pooling.Parameters.Mode, mode);
                Assert.Equal(pooling.Parameters.Height, windowHeight);
                Assert.Equal(pooling.Parameters.Width, windowWidth);
                Assert.Equal(pooling.Parameters.HeightStride, verticalStride);
                Assert.Equal(pooling.Parameters.WidthStride, horizontalStride);
            }
        }

        public static IEnumerable<object[]> FiltersConfigurations
        {
            get
            {
                return new[]
                {    
                    new object[] { new CudnnFilterDescriptorParameters(CudnnType.Double, 100, 10, 5, 5) },                                                   
                    new object[] { new CudnnFilterDescriptorParameters(CudnnType.Float, 10, 100, 5, 5) },                                                   
                };
            }
        }

        [Theory, MemberData("FiltersConfigurations")]
        public void Cudnn_Descriptors_ConstructFilterWithSetup(CudnnFilterDescriptorParameters param)
        {
            using (var filter = CudnnContext.CreateFilter(param))
            {
                Assert.True(filter.IsInitialized);

                CudnnType dataType = default(CudnnType);
                int k = 0, c = 0, h = 0, w = 0;
                CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetFilterDescriptor(filter.Handle, out dataType, out k, out c, out h, out w));

                Assert.Equal(filter.Parameters.Type, dataType);
                Assert.Equal(filter.Parameters.Output, k);
                Assert.Equal(filter.Parameters.Input, c);
                Assert.Equal(filter.Parameters.Height, h);
                Assert.Equal(filter.Parameters.Width, w);                
            }
        }
    }
}
