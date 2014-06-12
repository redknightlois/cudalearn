using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class GpuMatrix<T> : IEquatable<GpuMatrix<T>>, IGpuMatrixStorage<T>  where T : struct
    {
        public readonly int Rows;
        public readonly int Columns;

        /// <summary>
        /// Data is stored in Column-Major to be compliant with the BLAS packages
        /// </summary>
        private readonly CudaDeviceVariable<T> GpuData;

        public static readonly T Zero;
        public static readonly T One;
        public static readonly T DefaultEpsilon;

        public static readonly T Epsilon;

        static GpuMatrix()
        {
            if (typeof(T) == typeof(int))
            {
                Zero = (T)(object)0;
                One = (T)(object)1;
                DefaultEpsilon = (T)(object)0;                
            }
            else if (typeof(T) == typeof(float))
            {
                Zero = (T)(object)0f;
                One = (T)(object)1f;
                DefaultEpsilon = (T)(object)0.00001f;        
            }
            else if (typeof(T) == typeof(double))
            {
                Zero = (T)(object)0d;
                One = (T)(object)1d;
                DefaultEpsilon = (T)(object)0.00001d;               
            }
            else
            {
                Zero = default(T);
                One = default(T);
                DefaultEpsilon = default(T);   
            }

            Epsilon = DefaultEpsilon;            
        }

        public GpuMatrix(int iRows, int iCols)         // Matrix Class constructor
        {
            Contract.Requires<NotSupportedException>(typeof(T) == typeof(int) || typeof(T) == typeof(float) || typeof(T) == typeof(double));
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            Rows = iRows;
            Columns = iCols;
            GpuData = new CudaDeviceVariable<T>(Rows * Columns);            
        }

        public GpuMatrix(int iRows, int iCols, T value)         // Matrix Class constructor
        {            
            Contract.Requires<NotSupportedException>(typeof(T) == typeof(int) || typeof(T) == typeof(float) || typeof(T) == typeof(double));
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            Rows = iRows;
            Columns = iCols;
            GpuData = new CudaDeviceVariable<T>(Rows * Columns);

            for (int i = 0; i < Columns ; i++)
                for (int j = 0; j < Rows; j++)
                    this[i, j] = value;
        }

        public bool IsSquare
        {
            get { return Rows == Columns; }
        }
        public T this[int iRow, int iCol]      // Access this matrix as a 2D array
        {
            get
            {
                Contract.Requires(iRow >= 0 && iRow < Rows);
                Contract.Requires(iCol >= 0 && iCol < Columns);

                return GpuData[Rows * iCol + iRow];
            }
            set
            {
                Contract.Requires(iRow >= 0 && iRow < Rows);
                Contract.Requires(iCol >= 0 && iCol < Columns);

                if ( !CudaLearnModule.AllowHandyForDebugButVerySlowGpuMemoryAccess )
                    throw new InvalidOperationException("You cannot set values individually in the GPUMatrix<T> for performance reasons.");

                GpuData[Rows * iCol + iRow] = value;
            }
        }

        /// <summary>
        /// Creates a zero matrix
        /// </summary>
        /// <param name="iRows"></param>
        /// <param name="iCols"></param>
        /// <returns></returns>
        public static GpuMatrix<T> Zeroes(int iRows, int iCols)
        {
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            return new GpuMatrix<T>(iRows, iCols);
        }

        /// <summary>
        /// Creates an identity matrix.
        /// </summary>
        public static GpuMatrix<T> Identity(int iRows, int iCols)
        {
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            var matrix = new GpuMatrix<T>(iRows, iCols);

            for (int i = 0; i < Math.Min(iRows, iCols); i++)
                matrix[i, i] = GpuMatrix<T>.One;
            return matrix;
        }

        /// <summary>
        /// Matrix transpose, for any rectangular matrix
        /// </summary>
        public static GpuMatrix<T> Transpose(GpuMatrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            throw new NotImplementedException();
        }

        /// <summary>
        /// Matrix transpose, for any rectangular matrix
        /// </summary>
        public GpuMatrix<T> Transpose()
        {
            return GpuMatrix<T>.Transpose(this);
        }

        public GpuMatrix<T> Invert()
        {
            return GpuMatrix<T>.Invert(this);
        }

        public static GpuMatrix<T> Invert(GpuMatrix<T> m)
        {
            if (typeof(T) == typeof(float))
            {
                var t = m as GpuMatrix<float>;
                throw new NotImplementedException();
                //return MatrixHelper.Invert(t) as GpuMatrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as GpuMatrix<double>;
                throw new NotImplementedException();
                //return MatrixHelper.Invert(t) as GpuMatrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the GpuMatrix<T> class.");
        }


        public bool Equals(GpuMatrix<T> other)
        {
            if (other == null)
                return false;

            return Equals(this, other, GpuMatrix<T>.Epsilon);
        }

        public static bool Equals(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            return Equals(m1, m2, GpuMatrix<T>.Epsilon);
        }
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;

                // Suitable nullity checks etc, of course :)
                hash = hash * 23 + this.Rows.GetHashCode();
                hash = hash * 23 + this.Columns.GetHashCode();
                hash = hash * 23 + this.GpuData.GetHashCode();
                return hash;
            }
        }

        public static bool Equals(GpuMatrix<T> m1, GpuMatrix<T> m2, T epsilon)
        {
            if (m1 == null || m2 == null)
                return false;

            if (m1.Columns != m2.Columns || m1.Rows != m2.Rows)
                return false;

            if (typeof(T) == typeof(int))
            {
                var t1 = m1 as GpuMatrix<int>;
                var t2 = m2 as GpuMatrix<int>;
                var e = (int)(object)epsilon;
                //return MatrixHelper.Equals(t1, t2, e);
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = m1 as GpuMatrix<float>;
                var t2 = m2 as GpuMatrix<float>;
                var e = (float)(object)epsilon;
                //return MatrixHelper.Equals(t1, t2, e);
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as GpuMatrix<double>;
                var t2 = m2 as GpuMatrix<double>;
                var e = (double)(object)epsilon;
                //return MatrixHelper.Equals(t1, t2, e);
                throw new NotImplementedException();
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static GpuMatrix<T> operator *(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (typeof(T) == typeof(int))
            {
                var t1 = m1 as GpuMatrix<int>;
                var t2 = m2 as GpuMatrix<int>;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = m1 as GpuMatrix<float>;
                var t2 = m2 as GpuMatrix<float>;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as GpuMatrix<double>;
                var t2 = m2 as GpuMatrix<double>;
                throw new NotImplementedException();
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static GpuMatrix<T> operator *(T c, GpuMatrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var t = m as GpuMatrix<int>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as GpuMatrix<float>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as GpuMatrix<double>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static GpuMatrix<T> operator *(GpuMatrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c * m;
        }

        public static GpuMatrix<T> operator -(GpuMatrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var t = m as GpuMatrix<int>;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as GpuMatrix<float>;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as GpuMatrix<double>;
                throw new NotImplementedException();
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }
        public static GpuMatrix<T> operator +(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (typeof(T) == typeof(int))
            {
                var t1 = m1 as GpuMatrix<int>;
                var t2 = m2 as GpuMatrix<int>;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = m1 as GpuMatrix<float>;
                var t2 = m2 as GpuMatrix<float>;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as GpuMatrix<double>;
                var t2 = m2 as GpuMatrix<double>;
                throw new NotImplementedException();
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static GpuMatrix<T> operator -(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (typeof(T) == typeof(int))
            {
                var t1 = m1 as GpuMatrix<int>;
                var t2 = m2 as GpuMatrix<int>;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = m1 as GpuMatrix<float>;
                var t2 = m2 as GpuMatrix<float>;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as GpuMatrix<double>;
                var t2 = m2 as GpuMatrix<double>;
                throw new NotImplementedException();
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static GpuMatrix<T> operator +(T c, GpuMatrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var t = m as GpuMatrix<int>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as GpuMatrix<float>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as GpuMatrix<double>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static GpuMatrix<T> operator +(GpuMatrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c + m;
        }

        public static GpuMatrix<T> operator -(T c, GpuMatrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var t = m as GpuMatrix<int>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as GpuMatrix<float>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as GpuMatrix<double>;
                var c1 = (object)c;
                throw new NotImplementedException();
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static GpuMatrix<T> operator -(GpuMatrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c - m;
        }

        CudaDeviceVariable<T> IGpuMatrixStorage<T>.GetDeviceMemory()
        {
            return GpuData;
        }
    }
}
