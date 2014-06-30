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
    public static class Functions
    {
        public static Matrix<T> ElementwisePow<T>(this Matrix<T> m, T pow) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            int lenght = m.Rows * m.Columns;
            var target = new Matrix<T>(m.Rows, m.Columns);

            if (typeof(T) == typeof(int))
            {
                var mt1 = ((IHostMatrixStorage<int>)m).GetHostMemory();
                var t = ((IHostMatrixStorage<int>)target).GetHostMemory();
                var power = (int)(object)pow;

                for (int i = 0; i < lenght; i++)
                {
                    t[i] = (int)Math.Pow(mt1[i], power);
                }
            }
            else if (typeof(T) == typeof(float))
            {
                var mt1 = ((IHostMatrixStorage<float>)m).GetHostMemory();
                var t = ((IHostMatrixStorage<float>)target).GetHostMemory();
                var power = (float)(object)pow;

                for (int i = 0; i < lenght; i++)
                {
                    t[i] = (float)Math.Pow(mt1[i], power);
                }
            }
            else if (typeof(T) == typeof(double))
            {
                var mt1 = ((IHostMatrixStorage<double>)m).GetHostMemory();
                var t = ((IHostMatrixStorage<double>)target).GetHostMemory();
                var power = (double)(object)pow;

                for (int i = 0; i < lenght; i++)
                {
                    t[i] = Math.Pow(mt1[i], power);
                }
            }
            else throw new NotSupportedException("Type: {0} is not supported by the ElementWisePow<T> method.");

            return target;
        }

        public static GpuMatrix<T> ElementwisePow<T>(this GpuMatrix<T> m, T pow ) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            int numElements = m.Rows * m.Columns;
            var target = new GpuMatrix<T>(m.Rows, m.Columns);

            // arrayBinaryLess
            var mt = ((IGpuMatrixStorage<T>)m).GetDeviceMemory();            
            var t = ((IGpuMatrixStorage<T>)target).GetDeviceMemory();

            var context = CudaLearnModule.Context;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("math.ptx", GetCudaOperationWithSuffix<T>("mPow"));
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            kernel.Run(mt.DevicePointer, pow, t.DevicePointer, numElements);

            return target;
        }

        public static T Mean<T>(this Matrix<T> m) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var m1 = m as Matrix<int>;
                return (T)(object)(m1.Sum() / (m.Rows * m.Columns));
            }
            else if (typeof(T) == typeof(float))
            {
                var m1 = m as Matrix<float>;
                return (T)(object)(m1.Sum() / (m.Rows * m.Columns));

            }
            else if (typeof(T) == typeof(double))
            {
                var m1 = m as Matrix<double>;
                return (T)(object)(m1.Sum() / (m.Rows * m.Columns));
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Mean<T> method.");
        }

        public static T Mean<T>(this GpuMatrix<T> m) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            T summation = Sum(m);

            if (typeof(T) == typeof(int))
            {
                var value = (int)(object)summation;
                return (T)(object)(value / (m.Rows * m.Columns));
            }
            else if (typeof(T) == typeof(float))
            {
                var value = (float)(object)summation;
                return (T)(object)(value / (m.Rows * m.Columns));
            }
            else if (typeof(T) == typeof(double))
            {
                var value = (double)(object)summation;
                return (T)(object)(value / (m.Rows * m.Columns));
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Mean<T> method.");
        }

        public static T Sum<T> ( this GpuMatrix<T> m ) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            var sumRows = Sum(m, Axis.Rows);
            var sum = Sum(sumRows, Axis.Columns);

            using (var scope = new MemoryAccessScope<T>(sum))
            {
                return scope[0, 0];
            }
        }

        public static T Sum<T>(this Matrix<T> m) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            int lenght = m.Rows * m.Columns;
            if (typeof(T) == typeof(int))
            {
                var mt = ((IHostMatrixStorage<int>)m).GetHostMemory();

                int result = 0;
                for (int i = 0; i < lenght; i++)
                    result += mt[i];
                return (T)(object)result;
            }
            else if (typeof(T) == typeof(float))
            {
                var mt = ((IHostMatrixStorage<float>)m).GetHostMemory();

                float result = 0;
                for (int i = 0; i < lenght; i++)
                    result += mt[i];
                return (T)(object)result;
            }
            else if (typeof(T) == typeof(double))
            {
                var mt = ((IHostMatrixStorage<double>)m).GetHostMemory();

                double result = 0;
                for (int i = 0; i < lenght; i++)
                    result += mt[i];
                return (T)(object)result;
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Sum<T> method.");
        }

        public static Matrix<T> Sum<T>(this Matrix<T> m, Axis axis) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);
            Contract.Requires<ArgumentException>(axis != Axis.None);

            int items = axis == Axis.Columns ? m.Rows : m.Columns;
            int majorStride = axis == Axis.Columns ? m.Rows : 1;
            int minorStride = axis == Axis.Columns ? 1 : m.Rows;
            Matrix<T> target = new Matrix<T>(axis == Axis.Columns ? 1 : m.Rows, axis == Axis.Columns ? m.Columns : 1, Matrix<T>.Zero);

            if (typeof(T) == typeof(int))
            {
                var mt = ((IHostMatrixStorage<int>)m).GetHostMemory();
                var t = ((IHostMatrixStorage<int>)target).GetHostMemory();

                int ptr = 0;
                for (int majorItem = 0; majorItem < t.Length; majorItem++)
                {
                    for (int i = 0; i < items; i++)
                        t[majorItem] += mt[ptr + i * minorStride];

                    ptr += majorStride;
                }
            }
            else if (typeof(T) == typeof(float))
            {
                var mt = ((IHostMatrixStorage<float>)m).GetHostMemory();
                var t = ((IHostMatrixStorage<float>)target).GetHostMemory();

                int ptr = 0;
                for (int majorItem = 0; majorItem < t.Length; majorItem++)
                {
                    for (int i = 0; i < items; i++)
                        t[majorItem] += mt[ptr + i * minorStride];

                    ptr += majorStride;
                }
            }
            else if (typeof(T) == typeof(double))
            {
                var mt = ((IHostMatrixStorage<double>)m).GetHostMemory();
                var t = ((IHostMatrixStorage<double>)target).GetHostMemory();

                int ptr = 0;
                for (int majorItem = 0; majorItem < t.Length; majorItem++)
                {
                    for (int i = 0; i < items; i++)
                        t[majorItem] += mt[ptr + i * minorStride];

                    ptr += majorStride;
                }               
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Sum<T> method.");

            return target;
        }

        public static GpuMatrix<T> Sum<T>(this GpuMatrix<T> m, Axis axis) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);
            Contract.Requires<ArgumentException>(axis != Axis.None);

            if ( axis == Axis.Rows )
            {
                var target = new GpuMatrix<T>(m.Columns, 1, GpuMatrix<T>.One);
                return m * target;
            }
            else
            {
                var target = new GpuMatrix<T>(1, m.Rows, GpuMatrix<T>.One);
                return target * m;
            }
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

        public static T Dot<T>(this Matrix<T> m1, Matrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Rows == m2.Rows && m1.Columns == m2.Columns);
            Contract.Requires<ArgumentException>(m1.Rows == 1 || m1.Columns == 1);

            int lenght = m1.Rows * m1.Columns;
            if (typeof(T) == typeof(int))
            {
                var mt1 = ((IHostMatrixStorage<int>)m1).GetHostMemory();
                var mt2 = ((IHostMatrixStorage<int>)m2).GetHostMemory();

                int result = 0;
                for (int i = 0; i < lenght; i++)
                    result += mt1[i] * mt2[i];
                return (T)(object)result;
            }
            else if (typeof(T) == typeof(float))
            {
                var mt1 = ((IHostMatrixStorage<float>)m1).GetHostMemory();
                var mt2 = ((IHostMatrixStorage<float>)m2).GetHostMemory();

                float result = 0;
                for (int i = 0; i < lenght; i++)
                    result += mt1[i] * mt2[i];
                return (T)(object)result;
            }
            else if (typeof(T) == typeof(double))
            {
                var mt1 = ((IHostMatrixStorage<double>)m1).GetHostMemory();
                var mt2 = ((IHostMatrixStorage<double>)m2).GetHostMemory();

                double result = 0;
                for (int i = 0; i < lenght; i++)
                    result += mt1[i] * mt2[i];
                return (T)(object)result;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static T Dot<T>(this GpuMatrix<T> m1, GpuMatrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Rows == m2.Rows && m1.Columns == m2.Columns);
            Contract.Requires<ArgumentException>(m1.Rows == 1 || m1.Columns == 1);

            throw new NotImplementedException();
        }

        public static Matrix<T> Exp<T>(this Matrix<T> m) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            var target = new Matrix<T>(m.Rows, m.Columns);

            int lenght = m.Rows * m.Columns;

            if (typeof(T) == typeof(float))
            {
                var mt = ((IHostMatrixStorage<float>)m).GetHostMemory();
                var t = ((IHostMatrixStorage<float>)target).GetHostMemory();

                for (int i = 0; i < lenght; i++)
                    t[i] = (float)Math.Exp(mt[i]);
            }
            else if (typeof(T) == typeof(double))
            {
                var mt = ((IHostMatrixStorage<double>)m).GetHostMemory();
                var t = ((IHostMatrixStorage<double>)target).GetHostMemory();

                for (int i = 0; i < lenght; i++)
                    t[i] = Math.Exp(mt[i]);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");

            return target;
        }

        public static GpuMatrix<T> Exp<T>(this GpuMatrix<T> m) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            var mt = ((IGpuMatrixStorage<T>)m).GetDeviceMemory();
            var target = new GpuMatrix<T>(m.Rows, m.Columns);
            var t = ((IGpuMatrixStorage<T>)target).GetDeviceMemory();

            int numElements = m.Rows * m.Columns;

            var context = CudaLearnModule.Context;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("math.ptx", GetCudaOperationWithSuffix<T>("mExp"));
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            kernel.Run(mt.DevicePointer, t.DevicePointer, numElements);

            return target;
        }

        public static Matrix<T> BinaryLess<T>(this Matrix<T> m1, Matrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Rows == m2.Rows && m1.Columns == m2.Columns);

            int lenght = m1.Rows * m1.Columns;
            var target = new Matrix<T>(m1.Rows, m1.Columns);

            if (typeof(T) == typeof(int))
            {
                var mt1 = ((IHostMatrixStorage<int>)m1).GetHostMemory();
                var mt2 = ((IHostMatrixStorage<int>)m2).GetHostMemory();
                var t = ((IHostMatrixStorage<int>)target).GetHostMemory();

                for (int i = 0; i < lenght; i++)
                {
                    t[i] = mt1[i] < mt2[i] ? Matrix<int>.One : Matrix<int>.Zero;
                }
            }
            else if (typeof(T) == typeof(float))
            {
                var mt1 = ((IHostMatrixStorage<float>)m1).GetHostMemory();
                var mt2 = ((IHostMatrixStorage<float>)m2).GetHostMemory();
                var t = ((IHostMatrixStorage<float>)target).GetHostMemory();

                for (int i = 0; i < lenght; i++)
                {
                    t[i] = mt1[i] < mt2[i] ? Matrix<float>.One : Matrix<float>.Zero;
                }
            }
            else if (typeof(T) == typeof(double))
            {
                var mt1 = ((IHostMatrixStorage<double>)m1).GetHostMemory();
                var mt2 = ((IHostMatrixStorage<double>)m2).GetHostMemory();
                var t = ((IHostMatrixStorage<double>)target).GetHostMemory();

                for (int i = 0; i < lenght; i++)
                {
                    t[i] = mt1[i] < mt2[i] ? Matrix<double>.One : Matrix<double>.Zero;
                }
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");

            return target;
        }

        public static GpuMatrix<T> BinaryLess<T>(this GpuMatrix<T> m1, GpuMatrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);
            Contract.Requires<ArgumentException>(m1.Rows == m2.Rows && m1.Columns == m2.Columns);

            int numElements = m1.Rows * m1.Columns;
            var target = new GpuMatrix<T>(m1.Rows, m1.Columns);

            // arrayBinaryLess
            var m1t = ((IGpuMatrixStorage<T>)m1).GetDeviceMemory();
            var m2t = ((IGpuMatrixStorage<T>)m2).GetDeviceMemory();
            var t = ((IGpuMatrixStorage<T>)target).GetDeviceMemory();

            var context = CudaLearnModule.Context;
            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            CudaKernel kernel = context.LoadKernelPTX("arrayOperations.ptx", GetCudaOperationWithSuffix<T>("arrayBinaryLess"));
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);

            kernel.Run(m1t.DevicePointer, m2t.DevicePointer, t.DevicePointer, numElements);

            return target;
        }
    }
}
