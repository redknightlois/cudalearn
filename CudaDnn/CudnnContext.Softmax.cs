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

        public void Forward(CudnnSoftmaxAlgorithm algorithm, CudnnSoftmaxMode mode,
                            CudnnTensorDescriptor srcTensor, CudaDeviceVariable<float> srcData, CudnnTensorDescriptor destTensor, CudaDeviceVariable<float> destData)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            ThrowIfNotInitialized();
            CheckIfCompatible(CudnnType.Float, srcTensor, destTensor);

            Invoke(() => CudnnNativeMethods.cudnnSoftmaxForward(handle, algorithm, mode,
                                                    srcTensor.Handle, srcData.DevicePointer, destTensor.Handle, destData.DevicePointer));
        }

        public void Forward(CudnnSoftmaxAlgorithm algorithm, CudnnSoftmaxMode mode,
                            CudnnTensorDescriptor srcTensor, CudaDeviceVariable<double> srcData, CudnnTensorDescriptor destTensor, CudaDeviceVariable<double> destData)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            ThrowIfNotInitialized();
            CheckIfCompatible(CudnnType.Double, srcTensor, destTensor);

            Invoke(() => CudnnNativeMethods.cudnnSoftmaxForward(handle, algorithm, mode,
                                                    srcTensor.Handle, srcData.DevicePointer, destTensor.Handle, destData.DevicePointer));
        }

        public void Backward(CudnnSoftmaxAlgorithm algorithm, CudnnSoftmaxMode mode,
                             CudnnTensorDescriptor srcTensor, CudaDeviceVariable<float> srcData, CudnnTensorDescriptor srcDiffTensor, CudaDeviceVariable<float> srcDiffData,
                             CudnnTensorDescriptor destDiffTensor, CudaDeviceVariable<float> destDiffData)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(srcDiffTensor != null);
            Contract.Requires(srcDiffData != null);
            Contract.Requires(destDiffTensor != null);
            Contract.Requires(destDiffData != null);

            ThrowIfNotInitialized();
            CheckIfCompatible(CudnnType.Float, srcTensor, srcDiffTensor, destDiffTensor);

            Invoke(() => CudnnNativeMethods.cudnnSoftmaxBackward(handle, algorithm, mode,
                                                     srcTensor.Handle, srcData.DevicePointer, srcDiffTensor.Handle, srcDiffData.DevicePointer,
                                                     destDiffTensor.Handle, destDiffData.DevicePointer));
        }

        public void Backward(CudnnSoftmaxAlgorithm algorithm, CudnnSoftmaxMode mode,
                             CudnnTensorDescriptor srcTensor, CudaDeviceVariable<double> srcData, CudnnTensorDescriptor srcDiffTensor, CudaDeviceVariable<double> srcDiffData,
                             CudnnTensorDescriptor destDiffTensor, CudaDeviceVariable<double> destDiffData)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(srcDiffTensor != null);
            Contract.Requires(srcDiffData != null);
            Contract.Requires(destDiffTensor != null);
            Contract.Requires(destDiffData != null);

            ThrowIfNotInitialized();
            CheckIfCompatible(CudnnType.Double, srcTensor, srcDiffTensor, destDiffTensor);

            Invoke(() => CudnnNativeMethods.cudnnSoftmaxBackward(handle, algorithm, mode,
                                                     srcTensor.Handle, srcData.DevicePointer, srcDiffTensor.Handle, srcDiffData.DevicePointer,
                                                     destDiffTensor.Handle, destDiffData.DevicePointer));
        }

        public void Forward(CudnnSoftmaxAlgorithm algorithm, CudnnSoftmaxMode mode,
                            CudnnTensorDescriptor srcTensor, float[] srcData, CudnnTensorDescriptor destTensor, float[] destData)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            ThrowIfNotInitialized();
            CheckIfCompatible(CudnnType.Float, srcTensor, destTensor);

            using (var srcDataGpu = new CudaDeviceVariable<float>(srcData.Length))
            using (var destDataGpu = new CudaDeviceVariable<float>(destData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);

                Invoke(() => CudnnNativeMethods.cudnnSoftmaxForward(handle, algorithm, mode,
                                                                    srcTensor.Handle, srcDataGpu.DevicePointer, destTensor.Handle, destDataGpu.DevicePointer));
                destDataGpu.CopyToHost(destData);
            }
        }

        public void Forward(CudnnSoftmaxAlgorithm algorithm, CudnnSoftmaxMode mode,
                            CudnnTensorDescriptor srcTensor, double[] srcData, CudnnTensorDescriptor destTensor, double[] destData)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(destTensor != null);
            Contract.Requires(destData != null);

            ThrowIfNotInitialized();
            CheckIfCompatible(CudnnType.Double, srcTensor, destTensor);

            using (var srcDataGpu = new CudaDeviceVariable<double>(srcData.Length))
            using (var destDataGpu = new CudaDeviceVariable<double>(destData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);

                Invoke(() => CudnnNativeMethods.cudnnSoftmaxForward(handle, algorithm, mode,
                                                                    srcTensor.Handle, srcDataGpu.DevicePointer, destTensor.Handle, destDataGpu.DevicePointer));
                destDataGpu.CopyToHost(destData);
            }
        }

        public void Backward(CudnnSoftmaxAlgorithm algorithm, CudnnSoftmaxMode mode,
                             CudnnTensorDescriptor srcTensor, float[] srcData, CudnnTensorDescriptor srcDiffTensor, float[] srcDiffData,
                             CudnnTensorDescriptor destDiffTensor, float[] destDiffData)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(srcDiffTensor != null);
            Contract.Requires(srcDiffData != null);
            Contract.Requires(destDiffTensor != null);
            Contract.Requires(destDiffData != null);

            ThrowIfNotInitialized();
            CheckIfCompatible(CudnnType.Float, srcTensor, srcDiffTensor, destDiffTensor);

            using (var srcDataGpu = new CudaDeviceVariable<float>(srcData.Length))
            using (var srcDiffDataGpu = new CudaDeviceVariable<float>(srcDiffData.Length))
            using (var destDiffDataGpu = new CudaDeviceVariable<float>(destDiffData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);
                srcDiffDataGpu.CopyToDevice(srcDiffData);

                Invoke(() => CudnnNativeMethods.cudnnSoftmaxBackward(handle, algorithm, mode,
                                                                     srcTensor.Handle, srcDataGpu.DevicePointer, srcDiffTensor.Handle, srcDiffDataGpu.DevicePointer,
                                                                     destDiffTensor.Handle, destDiffDataGpu.DevicePointer));
                destDiffDataGpu.CopyToHost(destDiffData);
            }
        }

        public void Backward(CudnnSoftmaxAlgorithm algorithm, CudnnSoftmaxMode mode,
                             CudnnTensorDescriptor srcTensor, double[] srcData, CudnnTensorDescriptor srcDiffTensor, double[] srcDiffData,
                             CudnnTensorDescriptor destDiffTensor, double[] destDiffData)
        {
            Contract.Requires(srcTensor != null);
            Contract.Requires(srcData != null);
            Contract.Requires(srcDiffTensor != null);
            Contract.Requires(srcDiffData != null);
            Contract.Requires(destDiffTensor != null);
            Contract.Requires(destDiffData != null);

            ThrowIfNotInitialized();
            CheckIfCompatible(CudnnType.Double, srcTensor, srcDiffTensor, destDiffTensor);

            using (var srcDataGpu = new CudaDeviceVariable<double>(srcData.Length))
            using (var srcDiffDataGpu = new CudaDeviceVariable<double>(srcDiffData.Length))
            using (var destDiffDataGpu = new CudaDeviceVariable<double>(destDiffData.Length))
            {
                srcDataGpu.CopyToDevice(srcData);
                srcDiffDataGpu.CopyToDevice(srcDiffData);

                Invoke(() => CudnnNativeMethods.cudnnSoftmaxBackward(handle, algorithm, mode,
                                                                     srcTensor.Handle, srcDataGpu.DevicePointer, srcDiffTensor.Handle, srcDiffDataGpu.DevicePointer,
                                                                     destDiffTensor.Handle, destDiffDataGpu.DevicePointer));
                destDiffDataGpu.CopyToHost(destDiffData);
            }
        }
    }
}
