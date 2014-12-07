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
        public void Forward(CudnnPoolingDescriptor pooling, CudnnTensorDescriptor srcTensor, CudaDeviceVariable<float> srcData, CudnnTensorDescriptor destTensor, CudaDeviceVariable<float> destData)
        {
            Contract.Requires(pooling != null);
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, destTensor);

            Invoke(() => CudnnNativeMethods.cudnnPoolingForward(handle, pooling.Handle, srcTensor.Handle, srcData.DevicePointer, destTensor.Handle, destData.DevicePointer));
        }

        public void Forward(CudnnPoolingDescriptor pooling, CudnnTensorDescriptor srcTensor, CudaDeviceVariable<double> srcData, CudnnTensorDescriptor destTensor, CudaDeviceVariable<double> destData)
        {
            Contract.Requires(pooling != null);
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, destTensor);

            Invoke(() => CudnnNativeMethods.cudnnPoolingForward(handle, pooling.Handle, srcTensor.Handle, srcData.DevicePointer, destTensor.Handle, destData.DevicePointer));
        }

        public void Backward(CudnnPoolingDescriptor pooling, CudnnTensorDescriptor srcTensor, CudaDeviceVariable<float> srcData, CudnnTensorDescriptor srcDiffTensor, CudaDeviceVariable<float> srcDiffData,
                                                             CudnnTensorDescriptor destTensor, CudaDeviceVariable<float> destData, CudnnTensorDescriptor destDiffTensor, CudaDeviceVariable<float> destDiffData)
        {
            Contract.Requires(pooling != null);
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);
            Contract.Requires(srcDiffTensor != null);
            Contract.Requires(srcDiffData != null);
            Contract.Requires(destDiffTensor != null);
            Contract.Requires(destDiffData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, srcDiffTensor, destTensor, destDiffTensor);

            Invoke(() => CudnnNativeMethods.cudnnPoolingBackward(handle, pooling.Handle,
                                                     srcTensor.Handle, srcData.DevicePointer, srcDiffTensor.Handle, srcDiffData.DevicePointer,
                                                     destTensor.Handle, destData.DevicePointer, destDiffTensor.Handle, destDiffData.DevicePointer));
        }

        public void Backward(CudnnPoolingDescriptor pooling, CudnnTensorDescriptor srcTensor, CudaDeviceVariable<double> srcData, CudnnTensorDescriptor srcDiffTensor, CudaDeviceVariable<double> srcDiffData,
                                                             CudnnTensorDescriptor destTensor, CudaDeviceVariable<double> destData, CudnnTensorDescriptor destDiffTensor, CudaDeviceVariable<double> destDiffData)
        {
            Contract.Requires(pooling != null);
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);
            Contract.Requires(srcDiffTensor != null);
            Contract.Requires(srcDiffData != null);
            Contract.Requires(destDiffTensor != null);
            Contract.Requires(destDiffData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, srcDiffTensor, destTensor, destDiffTensor);

            Invoke(() => CudnnNativeMethods.cudnnPoolingBackward(handle, pooling.Handle,
                                                     srcTensor.Handle, srcData.DevicePointer, srcDiffTensor.Handle, srcDiffData.DevicePointer,
                                                     destTensor.Handle, destData.DevicePointer, destDiffTensor.Handle, destDiffData.DevicePointer));
        }


        public void Forward(CudnnPoolingDescriptor pooling, CudnnTensorDescriptor srcTensor, float[] srcData, CudnnTensorDescriptor destTensor, float[] destData)
        {
            Contract.Requires(pooling != null);
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, destTensor);

            using (var srcDataGpu = new CudaDeviceVariable<float>(srcData.Length))
            using (var destDataGpu = new CudaDeviceVariable<float>(destData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);

                Invoke(() => CudnnNativeMethods.cudnnPoolingForward(handle, pooling.Handle, srcTensor.Handle, srcDataGpu.DevicePointer, destTensor.Handle, destDataGpu.DevicePointer));
                destDataGpu.CopyToHost(destData);
            }
        }

        public void Forward(CudnnPoolingDescriptor pooling, CudnnTensorDescriptor srcTensor, double[] srcData, CudnnTensorDescriptor destTensor, double[] destData)
        {
            Contract.Requires(pooling != null);
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, destTensor);

            using (var srcDataGpu = new CudaDeviceVariable<double>(srcData.Length))
            using (var destDataGpu = new CudaDeviceVariable<double>(destData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);

                Invoke(() => CudnnNativeMethods.cudnnPoolingForward(handle, pooling.Handle, srcTensor.Handle, srcDataGpu.DevicePointer, destTensor.Handle, destDataGpu.DevicePointer));
                destDataGpu.CopyToHost(destData);
            }
        }

        public void Backward(CudnnPoolingDescriptor pooling, CudnnTensorDescriptor srcTensor, float[] srcData, CudnnTensorDescriptor srcDiffTensor, float[] srcDiffData,
                                                             CudnnTensorDescriptor destTensor, float[] destData, CudnnTensorDescriptor destDiffTensor, float[] destDiffData)
        {
            Contract.Requires(pooling != null);
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);
            Contract.Requires(srcDiffTensor != null);
            Contract.Requires(srcDiffData != null);
            Contract.Requires(destDiffTensor != null);
            Contract.Requires(destDiffData != null);

            CheckIfCompatible(CudnnType.Float, srcTensor, srcDiffTensor, destTensor, destDiffTensor);

            using (var srcDataGpu = new CudaDeviceVariable<float>(srcData.Length))
            using (var srcDiffDataGpu = new CudaDeviceVariable<float>(srcDiffData.Length))
            using (var destDataGpu = new CudaDeviceVariable<float>(destData.Length))
            using (var destDiffDataGpu = new CudaDeviceVariable<float>(destDiffData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);
                srcDiffDataGpu.CopyToDevice(srcDiffData);
                destDataGpu.CopyToDevice(destData);

                Invoke(() => CudnnNativeMethods.cudnnPoolingBackward(handle, pooling.Handle,
                                                                     srcTensor.Handle, srcDataGpu.DevicePointer, srcDiffTensor.Handle, srcDiffDataGpu.DevicePointer,
                                                                     destTensor.Handle, destDataGpu.DevicePointer, destDiffTensor.Handle, destDiffDataGpu.DevicePointer));
                destDiffDataGpu.CopyToHost(destDiffData);
            }
        }

        public void Backward(CudnnPoolingDescriptor pooling, CudnnTensorDescriptor srcTensor, double[] srcData, CudnnTensorDescriptor srcDiffTensor, double[] srcDiffData,
                                                             CudnnTensorDescriptor destTensor, double[] destData, CudnnTensorDescriptor destDiffTensor, double[] destDiffData)
        {
            Contract.Requires(pooling != null);
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);
            Contract.Requires(srcDiffTensor != null);
            Contract.Requires(srcDiffData != null);
            Contract.Requires(destDiffTensor != null);
            Contract.Requires(destDiffData != null);

            CheckIfCompatible(CudnnType.Double, srcTensor, srcDiffTensor, destTensor, destDiffTensor);

            using (var srcDataGpu = new CudaDeviceVariable<double>(srcData.Length))
            using (var srcDiffDataGpu = new CudaDeviceVariable<double>(srcDiffData.Length))
            using (var destDataGpu = new CudaDeviceVariable<double>(destData.Length))
            using (var destDiffDataGpu = new CudaDeviceVariable<double>(destDiffData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);
                srcDiffDataGpu.CopyToDevice(srcDiffData);
                destDataGpu.CopyToDevice(destData);

                Invoke(() => CudnnNativeMethods.cudnnPoolingBackward(handle, pooling.Handle,
                                                                     srcTensor.Handle, srcDataGpu.DevicePointer, srcDiffTensor.Handle, srcDiffDataGpu.DevicePointer,
                                                                     destTensor.Handle, destDataGpu.DevicePointer, destDiffTensor.Handle, destDiffDataGpu.DevicePointer));
                destDiffDataGpu.CopyToHost(destDiffData);
            }
        }
    }
}
