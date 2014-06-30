using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    partial class Matrix<T> : IEquatable<Matrix<T>>, IHostMatrixStorage<T>, IDisposable
        where T : struct
    {
        /// <summary>
        /// Creates a zero matrix
        /// </summary>
        /// <param name="iRows"></param>
        /// <param name="iCols"></param>
        /// <returns></returns>
        public static Matrix<T> Zeroes(int iRows, int iCols)
        {
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            return new Matrix<T>(iRows, iCols);
        }

        /// <summary>
        /// Creates an identity matrix.
        /// </summary>
        public static Matrix<T> Identity(int iRows)
        {
            Contract.Requires<ArgumentException>(iRows > 0);

            var matrix = new Matrix<T>(iRows, iRows);

            for (int i = 0; i < iRows; i++)
                matrix[i, i] = Matrix<T>.One;
            return matrix;
        }

        public static Matrix<T> Uniform(int rows)
        {
            return Uniform(rows, Matrix<T>.Zero, Matrix<T>.One);
        }
        public static Matrix<T> Uniform(int rows, T min, T max)
        {
            Contract.Requires<ArgumentOutOfRangeException>(rows > 0);

            var matrix = new Matrix<T>(rows, 1);

            if (typeof(T) == typeof(int))
            {
                var storage = ((IHostMatrixStorage<int>)matrix).GetHostMemory();
                Uniform(storage, (int)(object)min, (int)(object)max);
            }
            else if (typeof(T) == typeof(float))
            {
                var storage = ((IHostMatrixStorage<float>)matrix).GetHostMemory();
                Uniform(storage, (float)(object)min, (float)(object)max);
            }
            else if (typeof(T) == typeof(double))
            {
                var storage = ((IHostMatrixStorage<double>)matrix).GetHostMemory();
                Uniform(storage, (double)(object)min, (double)(object)max);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");

            return matrix;
        }

        public static Matrix<T> Uniform(int rows, int columns)
        {
            return Uniform(rows, columns, Matrix<T>.Zero, Matrix<T>.One);
        }

        public static Matrix<T> Uniform(int rows, int columns, T min, T max)
        {
            Contract.Requires<ArgumentOutOfRangeException>(rows > 0 && columns > 0);

            var matrix = new Matrix<T>(rows, columns);

            if (typeof(T) == typeof(int))
            {
                var storage = ((IHostMatrixStorage<int>)matrix).GetHostMemory();
                Uniform(storage, (int)(object)min, (int)(object)max);
            }
            else if (typeof(T) == typeof(float))
            {
                var storage = ((IHostMatrixStorage<float>)matrix).GetHostMemory();
                Uniform(storage, (float)(object)min, (float)(object)max);
            }
            else if (typeof(T) == typeof(double))
            {
                var storage = ((IHostMatrixStorage<double>)matrix).GetHostMemory();
                Uniform(storage, (double)(object)min, (double)(object)max);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");

            return matrix;
        }

        private static void Uniform(int[] storage, int min, int max)
        {
            Contract.Requires(storage != null);
            Contract.Requires(min < max);

            var generator = ThreadLocalRandom.Instance;
            for (int i = 0; i < storage.Length; i++)
                storage[i] = (int)(generator.NextDouble() * (max - min) + min);
        }

        private static void Uniform(float[] storage, float min, float max)
        {
            Contract.Requires(storage != null);
            Contract.Requires(min < max);

            var generator = ThreadLocalRandom.Instance;
            for (int i = 0; i < storage.Length; i++)
                storage[i] = (float)(generator.NextDouble() * (max - min) + min);
        }

        private static void Uniform(double[] storage, double min, double max)
        {
            Contract.Requires(storage != null);
            Contract.Requires(min < max);

            var generator = ThreadLocalRandom.Instance;
            for (int i = 0; i < storage.Length; i++)
                storage[i] = (double)(generator.NextDouble() * (max - min) + min);
        }

        public static Matrix<T> Normal(int rows)
        {
            return Normal(rows, Matrix<T>.Zero, Matrix<T>.One);
        }

        public static Matrix<T> Normal(int rows, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(rows > 0);

            var matrix = new Matrix<T>(rows, 1);

            if (typeof(T) == typeof(int))
            {
                var storage = ((IHostMatrixStorage<int>)matrix).GetHostMemory();
                Normal(storage, (int)(object)mean, (int)(object)deviation);
            }
            else if (typeof(T) == typeof(float))
            {
                var storage = ((IHostMatrixStorage<float>)matrix).GetHostMemory();
                Normal(storage, (float)(object)mean, (float)(object)deviation);
            }
            else if (typeof(T) == typeof(double))
            {
                var storage = ((IHostMatrixStorage<double>)matrix).GetHostMemory();
                Normal(storage, (double)(object)mean, (double)(object)deviation);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");

            return matrix;
        }

        public static Matrix<T> Normal(int rows, int columns)
        {
            return Normal(rows, columns, Matrix<T>.Zero, Matrix<T>.One);
        }

        public static Matrix<T> Normal(int rows, int columns, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(rows > 0 && columns > 0);

            var matrix = new Matrix<T>(rows, columns);

            if (typeof(T) == typeof(int))
            {
                var storage = ((IHostMatrixStorage<int>)matrix).GetHostMemory();
                Normal(storage, (int)(object)mean, (int)(object)deviation);
            }
            else if (typeof(T) == typeof(float))
            {
                var storage = ((IHostMatrixStorage<float>)matrix).GetHostMemory();
                Normal(storage, (float)(object)mean, (float)(object)deviation);
            }
            else if (typeof(T) == typeof(double))
            {
                var storage = ((IHostMatrixStorage<double>)matrix).GetHostMemory();
                Normal(storage, (double)(object)mean, (double)(object)deviation);
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");

            return matrix;
        }

        private static void Normal(int[] storage, int mean, int deviation)
        {
            Contract.Requires(storage != null);

            var generator = ThreadLocalRandom.Instance;
            for (int i = 0; i < storage.Length; i++)
            {
                // Use Box-Muller algorithm
                double u1 = generator.NextDouble();
                double u2 = generator.NextDouble();
                double r = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;
                double value = r * Math.Sin(theta);

                storage[i] = (int)(mean + deviation * value);
            }
        }

        private static void Normal(float[] storage, float mean, float deviation)
        {
            Contract.Requires(storage != null);

            var generator = ThreadLocalRandom.Instance;
            for (int i = 0; i < storage.Length; i++)
            {
                // Use Box-Muller algorithm
                double u1 = generator.NextDouble();
                double u2 = generator.NextDouble();
                double r = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;
                double value = r * Math.Sin(theta);

                storage[i] = (float)(mean + deviation * value);
            }
        }

        private static void Normal(double[] storage, double mean, double deviation)
        {
            Contract.Requires(storage != null);

            var generator = ThreadLocalRandom.Instance;
            for (int i = 0; i < storage.Length; i++)
            {
                // Use Box-Muller algorithm
                double u1 = generator.NextDouble();
                double u2 = generator.NextDouble();
                double r = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;
                double value = r * Math.Sin(theta);

                storage[i] = mean + deviation * value;
            }
        }

        private static Matrix<T> AddVectorOnColumns(Matrix<T> m, Matrix<T> v)
        {
            Contract.Requires(m != null);
            Contract.Requires(v.Columns == 1);
            Contract.Requires(v.Rows == m.Rows);

            var target = new Matrix<T>(m.Rows, m.Columns);

            if (typeof(T) == typeof(int))
            {
                var t1 = target as Matrix<int>;
                var m1 = m as Matrix<int>;
                var v1 = ((IHostMatrixStorage<int>)v).GetHostMemory();

                for (int i = 0; i < target.Rows; i++)
                {
                    for (int j = 0; j < target.Columns; j++)
                        t1[i, j] = m1[i, j] + v1[i];
                }
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = target as Matrix<float>;
                var m1 = m as Matrix<float>;
                var v1 = ((IHostMatrixStorage<float>)v).GetHostMemory();

                for (int i = 0; i < target.Rows; i++)
                {
                    for (int j = 0; j < target.Columns; j++)
                        t1[i, j] = m1[i, j] + v1[i];
                }
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = target as Matrix<double>;
                var m1 = m as Matrix<double>;
                var v1 = ((IHostMatrixStorage<double>)v).GetHostMemory();

                for (int i = 0; i < target.Rows; i++)
                {
                    for (int j = 0; j < target.Columns; j++)
                        t1[i, j] = m1[i, j] + v1[i];
                }
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");

            return target;
        }

        private static Matrix<T> AddVectorOnRows(Matrix<T> m, Matrix<T> v)
        {
            Contract.Requires(m != null);
            Contract.Requires(v.Rows == 1);
            Contract.Requires(v.Columns == m.Columns);

            var target = new Matrix<T>(m.Rows, m.Columns);

            if (typeof(T) == typeof(int))
            {
                var t1 = target as Matrix<int>;
                var m1 = m as Matrix<int>;
                var v1 = ((IHostMatrixStorage<int>)v).GetHostMemory();

                for (int i = 0; i < target.Rows; i++)
                {
                    for (int j = 0; j < target.Columns; j++)
                        t1[i, j] = m1[i, j] + v1[j];
                }
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = target as Matrix<float>;
                var m1 = m as Matrix<float>;
                var v1 = ((IHostMatrixStorage<float>)v).GetHostMemory();

                for (int i = 0; i < target.Rows; i++)
                {
                    for (int j = 0; j < target.Columns; j++)
                        t1[i, j] = m1[i, j] + v1[j];
                }
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = target as Matrix<double>;
                var m1 = m as Matrix<double>;
                var v1 = ((IHostMatrixStorage<double>)v).GetHostMemory();

                for (int i = 0; i < target.Rows; i++)
                {
                    for (int j = 0; j < target.Columns; j++)
                        t1[i, j] = m1[i, j] + v1[j];
                }
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");

            return target;
        }
    }
}
