using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    partial class GpuMatrix<T> : IEquatable<GpuMatrix<T>>, IGpuMatrixStorage<T>, IDisposable where T : struct
    {

        /// <summary>
        /// Creates a zero matrix
        /// </summary>
        /// <param name="iRows"></param>
        /// <param name="iCols"></param>
        /// <returns></returns>
        public static GpuMatrix<T> Zeroes(int iRows, int iCols)
        {
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            return new GpuMatrix<T>(iRows, iCols, GpuMatrix<T>.Zero);
        }

        /// <summary>
        /// Creates an identity matrix.
        /// </summary>
        public static GpuMatrix<T> Identity(int iRows)
        {
            Contract.Requires<ArgumentException>(iRows > 0);

            var m = new GpuMatrix<T>(iRows, iRows);
            BlasMath.SetIdentity(ref m);

            return m;
        }

        public static GpuMatrix<T> Uniform(int sizeX)
        {
            return Uniform(sizeX, GpuMatrix<T>.Zero, GpuMatrix<T>.One);
        }

        public static GpuMatrix<T> Uniform(int sizeX, T min, T max)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0);
            return Uniform(sizeX, 1, min, max);
        }

        public static GpuMatrix<T> Uniform(int sizeX, int sizeY)
        {
            return Uniform(sizeX, sizeY, GpuMatrix<T>.Zero, GpuMatrix<T>.One);
        }

        public static GpuMatrix<T> Uniform(int sizeX, int sizeY, T min, T max)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0 && sizeY > 0);

            if (typeof(T) == typeof(float))
            {
                var target = new GpuMatrix<float>(sizeX, sizeY);
                var t = ((IGpuMatrixStorage<float>)target).GetDeviceMemory();

                float fMin = (float)(object)min;
                float fMax = (float)(object)max;

                CudaLearnModule.RandomContext.GenerateUniform(t);

                return (((fMax - fMin) * target) + fMin) as GpuMatrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var target = new GpuMatrix<double>(sizeX, sizeY);
                var t = ((IGpuMatrixStorage<double>)target).GetDeviceMemory();

                double dMin = (double)(object)min;
                double dMax = (double)(object)max;

                CudaLearnModule.RandomContext.GenerateUniform(t);

                return (((dMax - dMin) * target) + dMin) as GpuMatrix<T>;
            }
            else throw new NotSupportedException("Type: {0} is not supported by the GpuMatrix<T> class.");
        }

        public static GpuMatrix<T> Normal(int sizeX)
        {            
            return Normal(sizeX, GpuMatrix<T>.Zero, GpuMatrix<T>.One);
        }

        public static GpuMatrix<T> Normal(int sizeX, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0);

            return Normal(sizeX, 1, mean, deviation);
        }

        public static GpuMatrix<T> Normal(int sizeX, int sizeY)
        {
            return Normal(sizeX, sizeY, GpuMatrix<T>.Zero, GpuMatrix<T>.One);
        }

        public static GpuMatrix<T> Normal(int sizeX, int sizeY, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0 && sizeY > 0);

            if (typeof(T) == typeof(float))
            {
                var target = new GpuMatrix<float>(sizeX, sizeY);
                var t = ((IGpuMatrixStorage<float>)target).GetDeviceMemory();
                CudaLearnModule.RandomContext.GenerateNormal(t, (float)(object)mean, (float)(object)deviation);

                return target as GpuMatrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var target = new GpuMatrix<double>(sizeX, sizeY);
                var t = ((IGpuMatrixStorage<double>)target).GetDeviceMemory();
                CudaLearnModule.RandomContext.GenerateNormal(t, (double)(object)mean, (double)(object)deviation);

                return target as GpuMatrix<T>;
            }
            else throw new NotSupportedException("Type: {0} is not supported by the GpuMatrix<T> class.");
        }

        private static string GetCudaOperationWithSuffix<T>(string name)
        {
            if (typeof(T) == typeof(float))
                return name + "1f";
            else if (typeof(T) == typeof(double))
                return name + "1d";
            else if (typeof(T) == typeof(int))
                return name + "1i";

            throw new NotSupportedException("Type: {0} is not supported by the GpuMatrix<T> class.");
        }

        private static GpuMatrix<T> AddVectorOnColumns(GpuMatrix<T> m, T multiplier, GpuMatrix<T> v)
        {
            var mt = ((IGpuMatrixStorage<T>)m).GetDeviceMemory();
            var vt = ((IGpuMatrixStorage<T>)v).GetDeviceMemory();

            var target = new GpuMatrix<T>(m.Rows, m.Columns);
            var t = ((IGpuMatrixStorage<T>)target).GetDeviceMemory();

            int numElements = m.Rows * m.Columns;

            var context = CudaLearnModule.Context;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("matrixOperations.ptx", GetCudaOperationWithSuffix<T>("matrixAddColVector"));
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            kernel.Run(mt.DevicePointer, vt.DevicePointer, multiplier, t.DevicePointer, m.Columns, m.Rows);

            context.Synchronize();
            return target;
        }

        private static GpuMatrix<T> AddVectorOnRows(GpuMatrix<T> m, T multiplier, GpuMatrix<T> v)
        {
            var mt = ((IGpuMatrixStorage<T>)m).GetDeviceMemory();
            var vt = ((IGpuMatrixStorage<T>)v).GetDeviceMemory();

            var target = new GpuMatrix<T>(m.Rows, m.Columns);
            var t = ((IGpuMatrixStorage<T>)target).GetDeviceMemory();

            int numElements = m.Rows * m.Columns;

            var context = CudaLearnModule.Context;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("matrixOperations.ptx", GetCudaOperationWithSuffix<T>("matrixAddRowVector"));
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            kernel.Run(mt.DevicePointer, vt.DevicePointer, multiplier, t.DevicePointer, m.Columns, m.Rows);

            context.Synchronize();

            return target;
        }
    }
}
