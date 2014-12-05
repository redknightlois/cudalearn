using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaDnn.Tests
{
    public class CudnnUseCase : CudaDnnTestBase
    {

        [Fact]
        public void Cudnn_UseCase_ForwardConvolution_Double()
        {
            CudnnContext.DefaultType = CudnnType.Double;
            CudnnContext.DefaultTensorFormat = CudnnTensorFormat.MajorRow;

            using (var context = CudnnContext.Create())
            {
                // Set some options and tensor dimensions              
                int nInput = 100;
                int filtersIn = 10;
                int filtersOut = 8;
                int heightIn = 20;
                int widthIn = 20;
                int heightFilter = 5;
                int widthFilter = 5;
                int paddingHeight = 4;
                int paddingWeight = 4;
                int verticalStride = 1;
                int horizontalStride = 1;
                int upscalex = 1;
                int upscaley = 1;

                // Input Tensor Data
                double[] xData = new double[nInput * filtersIn * heightIn * widthIn];
                ContinuousUniform.Samples(xData, 0, 1);

                // Filter Tensor Data
                double[] filterData = new double[filtersOut * filtersIn * heightFilter * widthFilter];
                ContinuousUniform.Samples(filterData, 0, 1);

                // Descriptor for input
                var xTensor = CudnnContext.CreateTensor(new CudnnTensorDescriptorParameters(nInput, filtersIn, heightFilter, widthFilter));

                // Filter descriptor
                var filter = CudnnContext.CreateFilter(new CudnnFilterDescriptorParameters(filtersOut, filtersIn, heightFilter, widthFilter));

                // Convolution descriptor
                var convolution = CudnnContext.CreateConvolution(new CudnnConvolutionDescriptorParameters(CudnnConvolutionMode.CrossCorrelation, xTensor, filter, paddingHeight, paddingWeight, verticalStride, horizontalStride, upscalex, upscaley));
                var output = convolution.GetOutputTensor(CudnnConvolutionPath.Forward);

                // Output tensor
                var yTensor = CudnnContext.CreateTensor(new CudnnTensorDescriptorParameters(nInput, filtersOut, output.Height, output.Width));
                double[] yData = new double[nInput * filtersOut * output.Height * output.Width];

                // Perform convolution
                context.Forward(xTensor, xData, filter, filterData, convolution, yTensor, yData, CudnnAccumulateResult.DoNotAccumulate);

                // Clean up
                xTensor.Dispose();
                yTensor.Dispose();
                filter.Dispose();
                convolution.Dispose();
            }
        }

        [Fact]
        public void Cudnn_UseCase_ForwardConvolution_Float()
        {
            CudnnContext.DefaultType = CudnnType.Float;
            CudnnContext.DefaultTensorFormat = CudnnTensorFormat.MajorRow;

            using (var context = CudnnContext.Create())
            {
                // Set some options and tensor dimensions              
                int nInput = 100;
                int filtersIn = 10;
                int filtersOut = 8;
                int heightIn = 20;
                int widthIn = 20;
                int heightFilter = 5;
                int widthFilter = 5;
                int paddingHeight = 4;
                int paddingWeight = 4;
                int verticalStride = 1;
                int horizontalStride = 1;
                int upscalex = 1;
                int upscaley = 1;

                var distribution = new ContinuousUniform(0, 1);

                // Input Tensor Data
                Vector<float> xDataVector = Vector<float>.Build.Dense(nInput * filtersIn * heightIn * widthIn);
                xDataVector.MapInplace(x => (float)distribution.Sample(), Zeros.Include);

                // Filter Tensor Data
                Vector<float> filterData = Vector<float>.Build.Dense(filtersOut * filtersIn * heightFilter * widthFilter);
                filterData.MapInplace(x => (float)distribution.Sample(), Zeros.Include);

                // Descriptor for input
                var xTensor = CudnnContext.CreateTensor(new CudnnTensorDescriptorParameters(nInput, filtersIn, heightFilter, widthFilter));

                // Filter descriptor
                var filter = CudnnContext.CreateFilter(new CudnnFilterDescriptorParameters(filtersOut, filtersIn, heightFilter, widthFilter));

                // Convolution descriptor
                var convolution = CudnnContext.CreateConvolution(new CudnnConvolutionDescriptorParameters(CudnnConvolutionMode.CrossCorrelation, xTensor, filter, paddingHeight, paddingWeight, verticalStride, horizontalStride, upscalex, upscaley));
                var output = convolution.GetOutputTensor(CudnnConvolutionPath.Forward);

                // Output tensor
                var yTensor = CudnnContext.CreateTensor(new CudnnTensorDescriptorParameters(nInput, filtersOut, output.Height, output.Width));
                float[] yData = new float[nInput * filtersOut * output.Height * output.Width];

                // Perform convolution
                context.Forward(xTensor, xDataVector.ToArray(), filter, filterData.ToArray(), convolution, yTensor, yData, CudnnAccumulateResult.DoNotAccumulate);

                // Clean up
                xTensor.Dispose();
                yTensor.Dispose();
                filter.Dispose();
                convolution.Dispose();
            }
        }
    }
}
