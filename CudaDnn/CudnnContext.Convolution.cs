using CudaDnn.Impl;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaDnn
{
    public sealed partial class CudnnContext
    {
        public void Forward(CudnnTensorDescriptor srcTensor, CudaDeviceVariable<float> srcData, CudnnFilterDescriptor filter, CudaDeviceVariable<float> filterData, CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor destTensor, CudaDeviceVariable<float> destData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(filter != null);
            Contract.Requires(filterData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, destTensor, filter);

            Invoke(() => CudnnNativeMethods.cudnnConvolutionForward(handle, srcTensor.Handle, srcData.DevicePointer, filter.Handle, filterData.DevicePointer, convolution.Handle, destTensor.Handle, destData.DevicePointer, accumulate));
        }

        public void Forward(CudnnTensorDescriptor srcTensor, CudaDeviceVariable<double> srcData, CudnnFilterDescriptor filter, CudaDeviceVariable<double> filterData, CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor destTensor, CudaDeviceVariable<double> destData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(filter != null);
            Contract.Requires(filterData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, destTensor, filter);

            Invoke(() => CudnnNativeMethods.cudnnConvolutionForward(handle, srcTensor.Handle, srcData.DevicePointer, filter.Handle, filterData.DevicePointer, convolution.Handle, destTensor.Handle, destData.DevicePointer, accumulate));
        }

        public void BackwardBias(CudnnTensorDescriptor srcTensor, CudaDeviceVariable<float> srcData, CudnnTensorDescriptor destTensor, CudaDeviceVariable<float> destData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, destTensor);

            Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardBias(handle, srcTensor.Handle, srcData.DevicePointer, destTensor.Handle, destData.DevicePointer, accumulate));            
        }

        public void BackwardBias(CudnnTensorDescriptor srcTensor, CudaDeviceVariable<double> srcData, CudnnTensorDescriptor destTensor, CudaDeviceVariable<double> destData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, destTensor);

            Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardBias(handle, srcTensor.Handle, srcData.DevicePointer, destTensor.Handle, destData.DevicePointer, accumulate));
        }

        public void BackwardFilter(CudnnTensorDescriptor srcTensor, CudaDeviceVariable<float> srcData, CudnnTensorDescriptor diffTensor, CudaDeviceVariable<float> diffData, CudnnConvolutionDescriptor convolution, CudnnFilterDescriptor gradient, CudaDeviceVariable<float> gradientData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(diffTensor != null);
            Contract.Requires(diffData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(gradient != null);
            Contract.Requires(gradientData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, diffTensor, gradient);

            Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardFilter(handle, srcTensor.Handle, srcData.DevicePointer, diffTensor.Handle, diffData.DevicePointer, convolution.Handle, gradient.Handle, gradientData.DevicePointer, accumulate));
        }

        public void BackwardFilter(CudnnTensorDescriptor srcTensor, CudaDeviceVariable<double> srcData, CudnnTensorDescriptor diffTensor, CudaDeviceVariable<double> diffData, CudnnConvolutionDescriptor convolution, CudnnFilterDescriptor gradient, CudaDeviceVariable<double> gradientData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(diffTensor != null);
            Contract.Requires(diffData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(gradient != null);
            Contract.Requires(gradientData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, diffTensor, gradient);

            Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardFilter(handle, srcTensor.Handle, srcData.DevicePointer, diffTensor.Handle, diffData.DevicePointer, convolution.Handle, gradient.Handle, gradientData.DevicePointer, accumulate));
        }

        public void BackwardData(CudnnFilterDescriptor filter, CudaDeviceVariable<float> filterData, CudnnTensorDescriptor diffTensor, CudaDeviceVariable<float> diffData, CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor gradient, CudaDeviceVariable<float> gradientData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(filter != null);
            Contract.Requires(filterData != null);
            Contract.Requires(diffTensor != null);
            Contract.Requires(diffData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(gradient != null);
            Contract.Requires(gradientData != null);

            CheckIfCompatible(CudnnType.Float, filter, diffTensor, gradient);

            Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardData(handle, filter.Handle, filterData.DevicePointer, diffTensor.Handle, diffData.DevicePointer, convolution.Handle, gradient.Handle, gradientData.DevicePointer, accumulate));
        }

        public void BackwardData(CudnnFilterDescriptor filter, CudaDeviceVariable<double> filterData, CudnnTensorDescriptor diffTensor, CudaDeviceVariable<double> diffData, CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor gradient, CudaDeviceVariable<double> gradientData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(filter != null);
            Contract.Requires(filterData != null);
            Contract.Requires(diffTensor != null);
            Contract.Requires(diffData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(gradient != null);
            Contract.Requires(gradientData != null);

            CheckIfCompatible(CudnnType.Double, filter, diffTensor, gradient);

            Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardData(handle, filter.Handle, filterData.DevicePointer, diffTensor.Handle, diffData.DevicePointer, convolution.Handle, gradient.Handle, gradientData.DevicePointer, accumulate));
        }





        public void Forward(CudnnTensorDescriptor srcTensor, float[] srcData, CudnnFilterDescriptor filter, float[] filterData, CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor destTensor, float[] destData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(filter != null);
            Contract.Requires(filterData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, destTensor, filter);

            using (var srcDataGpu = new CudaDeviceVariable<float>(srcData.Length))
            using (var filterDataGpu = new CudaDeviceVariable<float>(filterData.Length))
            using (var destDataGpu = new CudaDeviceVariable<float>(destData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);
                filterDataGpu.CopyToDevice(filterData);

                Invoke(() => CudnnNativeMethods.cudnnConvolutionForward(handle, srcTensor.Handle, srcDataGpu.DevicePointer, filter.Handle, filterDataGpu.DevicePointer, convolution.Handle, destTensor.Handle, destDataGpu.DevicePointer, accumulate));
                destDataGpu.CopyToHost(destData);
            }
        }

        public void Forward(CudnnTensorDescriptor srcTensor, double[] srcData, CudnnFilterDescriptor filter, double[] filterData, CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor destTensor, double[] destData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(filter != null);
            Contract.Requires(filterData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, destTensor, filter);

            using (var srcDataGpu = new CudaDeviceVariable<double>(srcData.Length))
            using (var filterDataGpu = new CudaDeviceVariable<double>(filterData.Length))
            using (var destDataGpu = new CudaDeviceVariable<double>(destData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);
                filterDataGpu.CopyToDevice(filterData);

                Invoke(() => CudnnNativeMethods.cudnnConvolutionForward(handle, srcTensor.Handle, srcDataGpu.DevicePointer, filter.Handle, filterDataGpu.DevicePointer, convolution.Handle, destTensor.Handle, destDataGpu.DevicePointer, accumulate));
                destDataGpu.CopyToHost(destData);
            }
        }

        public void BackwardBias(CudnnTensorDescriptor srcTensor, float[] srcData, CudnnTensorDescriptor destTensor, float[] destData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, destTensor);

            using (var srcDataGpu = new CudaDeviceVariable<float>(srcData.Length))
            using (var destDataGpu = new CudaDeviceVariable<float>(destData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);

                Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardBias(handle, srcTensor.Handle, srcDataGpu.DevicePointer, destTensor.Handle, destDataGpu.DevicePointer, accumulate));
                destDataGpu.CopyToHost(destData);
            }
        }

        public void BackwardBias(CudnnTensorDescriptor srcTensor, double[] srcData, CudnnTensorDescriptor destTensor, double[] destData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, destTensor);

            using (var srcDataGpu = new CudaDeviceVariable<double>(srcData.Length))
            using (var destDataGpu = new CudaDeviceVariable<double>(destData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);

                Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardBias(handle, srcTensor.Handle, srcDataGpu.DevicePointer, destTensor.Handle, destDataGpu.DevicePointer, accumulate));
                destDataGpu.CopyToHost(destData);
            }
        }

        public void BackwardFilter(CudnnTensorDescriptor srcTensor, float[] srcData, CudnnTensorDescriptor diffTensor, float[] diffData, CudnnConvolutionDescriptor convolution, CudnnFilterDescriptor gradient, float[] gradientData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(diffTensor != null);
            Contract.Requires(diffData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(gradient != null);
            Contract.Requires(gradientData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, diffTensor, gradient);

            using (var srcDataGpu = new CudaDeviceVariable<float>(srcData.Length))
            using (var diffDataGpu = new CudaDeviceVariable<float>(diffData.Length))
            using (var gradientDataGpu = new CudaDeviceVariable<float>(gradientData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);
                diffDataGpu.CopyToDevice(diffData);

                Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardFilter(handle, srcTensor.Handle, srcDataGpu.DevicePointer, diffTensor.Handle, diffDataGpu.DevicePointer, convolution.Handle, gradient.Handle, gradientDataGpu.DevicePointer, accumulate));
                gradientDataGpu.CopyToHost(gradientData);
            }
        }

        public void BackwardFilter(CudnnTensorDescriptor srcTensor, double[] srcData, CudnnTensorDescriptor diffTensor, double[] diffData, CudnnConvolutionDescriptor convolution, CudnnFilterDescriptor gradient, double[] gradientData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(diffTensor != null);
            Contract.Requires(diffData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(gradient != null);
            Contract.Requires(gradientData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, diffTensor, gradient);

            using (var srcDataGpu = new CudaDeviceVariable<double>(srcData.Length))
            using (var diffDataGpu = new CudaDeviceVariable<double>(diffData.Length))
            using (var gradientDataGpu = new CudaDeviceVariable<double>(gradientData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);
                diffDataGpu.CopyToDevice(diffData);

                Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardFilter(handle, srcTensor.Handle, srcDataGpu.DevicePointer, diffTensor.Handle, diffDataGpu.DevicePointer, convolution.Handle, gradient.Handle, gradientDataGpu.DevicePointer, accumulate));
                gradientDataGpu.CopyToHost(gradientData);
            }
        }

        public void BackwardData(CudnnFilterDescriptor filter, float[] filterData, CudnnTensorDescriptor diffTensor, float[] diffData, CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor gradient, float[] gradientData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(filter != null);
            Contract.Requires(filterData != null);
            Contract.Requires(diffTensor != null);
            Contract.Requires(diffData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(gradient != null);
            Contract.Requires(gradientData != null);

            CheckIfCompatible(CudnnType.Float, filter, diffTensor, gradient);

            using (var filterDataGpu = new CudaDeviceVariable<float>(filterData.Length))
            using (var diffDataGpu = new CudaDeviceVariable<float>(diffData.Length))
            using (var gradientDataGpu = new CudaDeviceVariable<float>(gradientData.Length))
            {
                filterDataGpu.CopyToDevice(filterData);
                diffDataGpu.CopyToDevice(diffData);

                Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardData(handle, filter.Handle, filterDataGpu.DevicePointer, diffTensor.Handle, diffDataGpu.DevicePointer, convolution.Handle, gradient.Handle, gradientDataGpu.DevicePointer, accumulate));
                gradientDataGpu.CopyToHost(gradientData);
            }
        }

        public void BackwardData(CudnnFilterDescriptor filter, double[] filterData, CudnnTensorDescriptor diffTensor, double[] diffData, CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor gradient, double[] gradientData, CudnnAccumulateResult accumulate)
        {
            Contract.Requires(filter != null);
            Contract.Requires(filterData != null);
            Contract.Requires(diffTensor != null);
            Contract.Requires(diffData != null);
            Contract.Requires(convolution != null);
            Contract.Requires(gradient != null);
            Contract.Requires(gradientData != null);

            CheckIfCompatible(CudnnType.Double, filter, diffTensor, gradient);

            using (var filterDataGpu = new CudaDeviceVariable<double>(filterData.Length))
            using (var diffDataGpu = new CudaDeviceVariable<double>(diffData.Length))
            using (var gradientDataGpu = new CudaDeviceVariable<double>(gradientData.Length))
            {
                filterDataGpu.CopyToDevice(filterData);
                diffDataGpu.CopyToDevice(diffData);

                Invoke(() => CudnnNativeMethods.cudnnConvolutionBackwardData(handle, filter.Handle, filterDataGpu.DevicePointer, diffTensor.Handle, diffDataGpu.DevicePointer, convolution.Handle, gradient.Handle, gradientDataGpu.DevicePointer, accumulate));

                gradientDataGpu.CopyToHost(gradientData);
            }
        }

    }
}
