using ManagedCuda;
using ManagedCuda.CudaBlas;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class BlasMath 
    {
        public static double SumMagnitudes<T>(GpuMatrix<T> matrix) where T : struct
        {
            Contract.Requires<ArgumentNullException>(matrix != null);
            Contract.Requires(matrix.Rows == 1);

            var storage = (IGpuMatrixStorage<T>)matrix;
            var blas = CudaLearnModule.BlasContext;
            var data = storage.GetDeviceMemory();

            if (typeof(T) == typeof(float))
            {
                return blas.AbsoluteSum(data as CudaDeviceVariable<float>, matrix.Rows);
            }
            else if (typeof(T) == typeof(double))
            {
                return blas.AbsoluteSum(data as CudaDeviceVariable<double>, matrix.Rows);
            }

            throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        public static double AbsoluteSum<T>(Matrix<T> matrix) where T : struct
        {
            Contract.Requires<ArgumentNullException>(matrix != null);
            Contract.Requires(matrix.Rows == 1);

            throw new NotImplementedException();
        }

        public static GpuMatrix<T> Axpb<T>(GpuMatrix<T> m1, T a, T b ) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);

            var context = CudaLearnModule.Context;
            var blas = CudaLearnModule.BlasContext;
            var d1 = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();

            var m3 = new GpuMatrix<T>(m1.Rows, m1.Columns);
            var d3 = ((IGpuMatrixStorage<T>)m3).GetDeviceMemory();

            int numElements = m1.Rows * m1.Columns;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("vectorOperations.ptx", GetCudaOperationWithSuffix<T>("vectorAxpb"));
            // TODO Minimize the tail effect. http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-minimize-the-tail-effect/
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);
            Console.WriteLine(string.Format("X:{0} Y:{1} Z:{2}", kernel.GridDimensions.x, kernel.GridDimensions.y, kernel.GridDimensions.z));

            if (typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int))
            {
                kernel.Run(d1.DevicePointer, a, b, d3.DevicePointer, numElements);
            }          
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");

            return m3;
        }

        public static GpuMatrix<T> Axpy<T>(GpuMatrix<T> m1, T alpha, GpuMatrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Rows == m2.Rows && m1.Columns == m2.Columns);

            var blas = CudaLearnModule.BlasContext;
            var d1 = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();
            var d2 = ((IGpuMatrixStorage<T>)m2).GetDeviceMemory();

            var m3 = new GpuMatrix<T>(m1.Rows, m1.Columns);
            var d3 = ((IGpuMatrixStorage<T>)m3).GetDeviceMemory();

            if (typeof(T) == typeof(float))
            {
                blas.Copy(d2 as CudaDeviceVariable<float>, 1, d3 as CudaDeviceVariable<float>, 1);
                blas.Axpy((float)(object)alpha, d1 as CudaDeviceVariable<float>, 1, d3 as CudaDeviceVariable<float>, 1);
                return m3;
            }
            else if (typeof(T) == typeof(double))
            {                
                blas.Copy(d2 as CudaDeviceVariable<double>, 1, d3 as CudaDeviceVariable<double>, 1);
                blas.Axpy((double)(object)alpha, d1 as CudaDeviceVariable<double>, 1, d3 as CudaDeviceVariable<double>, 1);
                return m3;
            }

            throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        public static void AxpyInPlace<T>(GpuMatrix<T> m1, T alpha, ref GpuMatrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Rows == m2.Rows && m1.Columns == m2.Columns);

            var blas = CudaLearnModule.BlasContext;
            var d1 = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();
            var d2 = ((IGpuMatrixStorage<T>)m2).GetDeviceMemory();

            if (typeof(T) == typeof(float))
            {
                blas.Axpy((float)(object)alpha, d1 as CudaDeviceVariable<float>, 1, d2 as CudaDeviceVariable<float>, 1);
            }
            else if (typeof(T) == typeof(double))
            {
                blas.Axpy((double)(object)alpha, d1 as CudaDeviceVariable<double>, 1, d2 as CudaDeviceVariable<double>, 1);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        public static GpuMatrix<T> Multiply<T>(GpuMatrix<T> m1, GpuMatrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Columns == m2.Rows);

            var blas = CudaLearnModule.BlasContext;
            var d1 = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();
            var d2 = ((IGpuMatrixStorage<T>)m2).GetDeviceMemory();

            var m3 = new GpuMatrix<T>(m1.Rows, m2.Columns);
            var d3 = ((IGpuMatrixStorage<T>)m3).GetDeviceMemory();

            if (typeof(T) == typeof(float))
            {
                blas.Gemm(Operation.NonTranspose, Operation.NonTranspose, m1.Rows, m2.Columns, m1.Columns, 1.0f, d1 as CudaDeviceVariable<float>, m1.Rows, d2 as CudaDeviceVariable<float>, m2.Rows, 0, d3 as CudaDeviceVariable<float>, m3.Rows);
                return m3;
            }
            else if (typeof(T) == typeof(double))
            {
                blas.Gemm(Operation.NonTranspose, Operation.NonTranspose, m1.Rows, m2.Columns, m1.Columns, 1.0d, d1 as CudaDeviceVariable<double>, m1.Rows, d2 as CudaDeviceVariable<double>, m2.Rows, 0, d3 as CudaDeviceVariable<double>, m3.Rows);
                return m3;
            }

            throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        public static bool Equals<T>(GpuMatrix<T> m1, GpuMatrix<T> m2) where T : struct
        {
            return Equals(m1, m2, Matrix<T>.Epsilon);
        }

        private static string GetCudaOperationWithSuffix<T>(string name)
        {
            if (typeof(T) == typeof(float))
                return name + "1f";
            else if (typeof(T) == typeof(double))
                return name + "1d";
            else if (typeof(T) == typeof(int))
                return name + "1i";

            throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }


        public static bool Equals<T>(GpuMatrix<T> m1, GpuMatrix<T> m2, T epsilon) where T : struct 
        {
            Contract.Requires<NotSupportedException>(typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int), "Type: {0} is not supported by the BLAS library.");
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);

            if (m1.Rows != m2.Rows || m1.Columns != m2.Columns)
                return false;

            var context = CudaLearnModule.Context;

            var d1 = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();
            var d2 = ((IGpuMatrixStorage<T>)m2).GetDeviceMemory();

            int numElements = m1.Rows * m1.Columns;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("vectorOperations.ptx", GetCudaOperationWithSuffix<T>("vectorEquals"));
            // TODO Minimize the tail effect. http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-minimize-the-tail-effect/
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);
            Console.WriteLine(string.Format("X:{0} Y:{1} Z:{2}", kernel.GridDimensions.x, kernel.GridDimensions.y, kernel.GridDimensions.z));

            var equals = new bool[1];
            CudaDeviceVariable<bool> result = new CudaDeviceVariable<bool>(1);
            kernel.Run(d1.DevicePointer, d2.DevicePointer, result.DevicePointer, numElements, epsilon);
            result.CopyToHost(equals);
            return equals[0];

        }
    }
}
