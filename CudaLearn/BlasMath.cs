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

        public static double SumMagnitudes<T>(Matrix<T> matrix) where T : struct
        {
            Contract.Requires<ArgumentNullException>(matrix != null);
            Contract.Requires(matrix.Rows == 1);

            throw new NotImplementedException();
        }

        #region Axpby = alpha * x + beta * y

        public static GpuMatrix<T> Axpby<T>(GpuMatrix<T> m1, T a, GpuMatrix<T> m2, T b) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Rows == m2.Rows && m1.Columns == m2.Columns);

            var context = CudaLearnModule.Context;
            var blas = CudaLearnModule.BlasContext;
            var d1 = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();
            var d2 = ((IGpuMatrixStorage<T>)m2).GetDeviceMemory();

            var m3 = new GpuMatrix<T>(m1.Rows, m1.Columns);
            var d3 = ((IGpuMatrixStorage<T>)m3).GetDeviceMemory();

            int numElements = m1.Rows * m1.Columns;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("vectorOperations.ptx", GetCudaOperationWithSuffix<T>("vectorAxpby"));
            // TODO Minimize the tail effect. http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-minimize-the-tail-effect/
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            if (typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int))
            {
                kernel.Run(d1.DevicePointer, a, d2.DevicePointer, b, d3.DevicePointer, numElements);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");

            return m3;
        }

        public static void AxpbyInPlace<T>(GpuMatrix<T> m1, T alpha, ref GpuMatrix<T> m2, T beta) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Rows == m2.Rows && m1.Columns == m2.Columns);

            var context = CudaLearnModule.Context;
            var blas = CudaLearnModule.BlasContext;
            var d1 = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();
            var d2 = ((IGpuMatrixStorage<T>)m2).GetDeviceMemory();

            int numElements = m1.Rows * m1.Columns;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("vectorOperations.ptx", GetCudaOperationWithSuffix<T>("vectorAxpby"));
            // TODO Minimize the tail effect. http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-minimize-the-tail-effect/
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            if (typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int))
            {
                kernel.Run(d1.DevicePointer, alpha, d2.DevicePointer, beta, d2.DevicePointer, numElements);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        #endregion

        #region Axpb = alpha * x + beta

        public static GpuMatrix<T> Axpb<T>(GpuMatrix<T> m1, T a, T b) where T : struct
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

            if (typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int))
            {
                kernel.Run(d1.DevicePointer, a, b, d3.DevicePointer, numElements);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");

            return m3;
        }

        public static void AxpbInPlace<T>(ref GpuMatrix<T> m1, T a, T b) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);

            var context = CudaLearnModule.Context;
            var blas = CudaLearnModule.BlasContext;
            var d1 = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();

            int numElements = m1.Rows * m1.Columns;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("vectorOperations.ptx", GetCudaOperationWithSuffix<T>("vectorAxpb"));
            // TODO Minimize the tail effect. http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-minimize-the-tail-effect/
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            if (typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int))
            {
                kernel.Run(d1.DevicePointer, a, b, d1.DevicePointer, numElements);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        public static void SetConstant<T>(ref GpuMatrix<T> m, T c) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            var context = CudaLearnModule.Context;
            var d1 = ((IGpuMatrixStorage<T>)m).GetDeviceMemory();

            int numElements = m.Rows * m.Columns;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("vectorOperations.ptx", GetCudaOperationWithSuffix<T>("vectorSet"));
            // TODO Minimize the tail effect. http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-minimize-the-tail-effect/
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            if (typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int))
            {
                kernel.Run(d1.DevicePointer, c, numElements);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");

        }

        public static void SetIdentity<T>( ref GpuMatrix<T> m ) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);
            Contract.Requires<ArgumentException>(m.Rows == m.Columns);

            var context = CudaLearnModule.Context;
            var d1 = ((IGpuMatrixStorage<T>)m).GetDeviceMemory();

            int threadsPerBlock = 8;
            int blocksPerGrid = (m.Rows + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("matrixOperations.ptx", GetCudaOperationWithSuffix<T>("matrixSetIdentity"));
            kernel.BlockDimensions = new dim3(threadsPerBlock, threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid, blocksPerGrid);

            if (typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int))
            {
                kernel.Run(d1.DevicePointer, m.Rows);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }


        #endregion

        #region Axpy = alpha * x + y

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

            }
            else if (typeof(T) == typeof(double))
            {
                blas.Copy(d2 as CudaDeviceVariable<double>, 1, d3 as CudaDeviceVariable<double>, 1);
                blas.Axpy((double)(object)alpha, d1 as CudaDeviceVariable<double>, 1, d3 as CudaDeviceVariable<double>, 1);

            }
            else if (typeof(T) == typeof(int))
            {
                int numElements = m1.Rows * m1.Columns;

                var context = CudaLearnModule.Context;
                int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
                int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

                CudaKernel kernel = context.LoadKernelPTX("vectorOperations.ptx", GetCudaOperationWithSuffix<T>("vectorAxpby"));
                // TODO Minimize the tail effect. http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-minimize-the-tail-effect/
                kernel.BlockDimensions = new dim3(threadsPerBlock);
                kernel.GridDimensions = new dim3(blocksPerGrid);

                kernel.Run(d1.DevicePointer, alpha, d2.DevicePointer, 1, d3.DevicePointer, numElements);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");

            return m3;            
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
            else if (typeof(T) == typeof(int))
            {
                int numElements = m1.Rows * m1.Columns;

                var context = CudaLearnModule.Context;
                int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
                int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

                CudaKernel kernel = context.LoadKernelPTX("vectorOperations.ptx", GetCudaOperationWithSuffix<T>("vectorAxpby"));
                // TODO Minimize the tail effect. http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-minimize-the-tail-effect/
                kernel.BlockDimensions = new dim3(threadsPerBlock);
                kernel.GridDimensions = new dim3(blocksPerGrid);

                kernel.Run(d1.DevicePointer, alpha, d2.DevicePointer, 1, d2.DevicePointer, numElements);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        #endregion 

        #region alpha * x * y + beta * z ,
        public static GpuMatrix<T> Gemm<T>(T alpha, GpuMatrix<T> x, GpuMatrix<T> y, T beta, GpuMatrix<T> z, BlasOperation opx = BlasOperation.NonTranspose, BlasOperation opy = BlasOperation.NonTranspose) where T : struct
        {
            Contract.Requires<ArgumentException>(opx == BlasOperation.NonTranspose ? x.Rows == z.Rows : x.Columns == z.Rows);
            Contract.Requires<ArgumentException>(opy == BlasOperation.NonTranspose ? y.Columns == z.Columns : y.Rows == z.Columns);

            var blas = CudaLearnModule.BlasContext;
            var x1 = ((IGpuMatrixStorage<T>)x).GetDeviceMemory();
            var y2 = ((IGpuMatrixStorage<T>)y).GetDeviceMemory();
            var z3 = ((IGpuMatrixStorage<T>)z).GetDeviceMemory();

            var o = new GpuMatrix<T>(opx == BlasOperation.NonTranspose ? x.Rows : x.Columns, opy == BlasOperation.NonTranspose ? y.Columns : y.Rows);
            var o1 = ((IGpuMatrixStorage<T>)o).GetDeviceMemory();

            int m = z.Rows;
            int n = z.Columns;
            int k = opy == BlasOperation.NonTranspose ? y.Rows : y.Columns ;

            if (typeof(T) == typeof(float))
            {
                float a = (float)(object)alpha;
                float b = (float)(object)beta;

                blas.Copy(z3 as CudaDeviceVariable<float>, 1, o1 as CudaDeviceVariable<float>, 1);
                blas.Gemm((Operation)opx, (Operation)opy, m, n, k, a, x1 as CudaDeviceVariable<float>, x.Rows, y2 as CudaDeviceVariable<float>, y.Rows, b, o1 as CudaDeviceVariable<float>, o.Rows);
                return o;
            }
            else if (typeof(T) == typeof(double))
            {
                double a = (double)(object)alpha;
                double b = (double)(object)beta;

                blas.Copy(z3 as CudaDeviceVariable<double>, 1, o1 as CudaDeviceVariable<double>, 1);
                blas.Gemm((Operation)opx, (Operation)opy, m, n, k, a, x1 as CudaDeviceVariable<double>, x.Rows, y2 as CudaDeviceVariable<double>, y.Rows, b, o1 as CudaDeviceVariable<double>, o.Rows);
                return o;
            }

            throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        public static void GemmInPlace<T>(T alpha, GpuMatrix<T> x, GpuMatrix<T> y, T beta, ref GpuMatrix<T> z, BlasOperation opx = BlasOperation.NonTranspose, BlasOperation opy = BlasOperation.NonTranspose) where T : struct
        {
            Contract.Requires<ArgumentException>(opx == BlasOperation.NonTranspose ? x.Rows == z.Rows : x.Columns == z.Rows);
            Contract.Requires<ArgumentException>(opy == BlasOperation.NonTranspose ? y.Columns == z.Columns : y.Rows == z.Columns);

            var blas = CudaLearnModule.BlasContext;
            var x1 = ((IGpuMatrixStorage<T>)x).GetDeviceMemory();
            var y2 = ((IGpuMatrixStorage<T>)y).GetDeviceMemory();
            var z3 = ((IGpuMatrixStorage<T>)z).GetDeviceMemory();

            int m = z.Rows;
            int n = z.Columns;
            int k = opy == BlasOperation.NonTranspose ? y.Rows : y.Columns;

            if (typeof(T) == typeof(float))
            {
                float a = (float)(object)alpha;
                float b = (float)(object)beta;

                blas.Gemm((Operation)opx, (Operation)opy,  m, n, k, a, x1 as CudaDeviceVariable<float>, x.Rows, y2 as CudaDeviceVariable<float>, y.Rows, b, z3 as CudaDeviceVariable<float>, z.Rows);
            }
            else if (typeof(T) == typeof(double))
            {
                double a = (double)(object)alpha;
                double b = (double)(object)beta;

                blas.Gemm((Operation)opx, (Operation)opy, m, n, k, a, x1 as CudaDeviceVariable<double>, x.Rows, y2 as CudaDeviceVariable<double>, y.Rows, b, z3 as CudaDeviceVariable<double>, z.Rows);
            }

            throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        #endregion

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

            var equals = new bool[1];
            CudaDeviceVariable<bool> result = new CudaDeviceVariable<bool>(1);
            kernel.Run(d1.DevicePointer, d2.DevicePointer, result.DevicePointer, numElements, epsilon);
            result.CopyToHost(equals);
            return equals[0];
        }
    }
}
