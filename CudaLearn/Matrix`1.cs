using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class Matrix<T> : IEquatable<Matrix<T>>
        where T : struct
    {                
        public readonly int Rows;
        public readonly int Columns;
        protected readonly T[,] Data;

        private Lazy<Decomposition<T>> decomposition;

        public Decomposition<T> Decomposition
        {
            get { return decomposition.Value; }
        }

        public static readonly T Zero;
        public static readonly T One;
        public static readonly T DefaultEpsilon;

        public static readonly T Epsilon;

        static Matrix ()
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

        public Matrix(int iRows, int iCols)         // Matrix Class constructor
        {
            Contract.Requires<NotSupportedException>(typeof(T) == typeof(int) || typeof(T) == typeof(float) || typeof(T) == typeof(double));
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            Rows = iRows;
            Columns = iCols;
            Data = new T[Rows, Columns];

            this.decomposition = new Lazy<Decomposition<T>>(() => MakeLU(this), true);
        }

        public Matrix(int iRows, int iCols, T value)         // Matrix Class constructor
        {            
            Contract.Requires<NotSupportedException>(typeof(T) == typeof(int) || typeof(T) == typeof(float) || typeof(T) == typeof(double));
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            Rows = iRows;
            Columns = iCols;
            Data = new T[Rows, Columns];

            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    Data[i, j] = value;

            this.decomposition = new Lazy<Decomposition<T>>(() => MakeLU(this), true);
        }

        public bool IsSquare
        {
            get { return Rows == Columns; }
        }

        public T this[int iRow, int iCol]      // Access this matrix as a 2D array
        {
            get { return Data[iRow, iCol]; }
            set { Data[iRow, iCol] = value; }
        }

        public Matrix<T> GetColumn(int k)
        {
            Matrix<T> m = new Matrix<T>(Rows, 1);
            for (int i = 0; i < Rows; i++)
                m[i, 0] = Data[i, k];
            return m;
        }

        public void SetColumn(Matrix<T> v, int k)
        {
            for (int i = 0; i < Rows; i++)
                Data[i, k] = v[i, 0];
        }

        public Matrix<T> GetRow(int k)
        {
            Matrix<T> m = new Matrix<T>(1, Columns);
            for (int i = 0; i < Columns; i++)
                m[0, i] = Data[k, i];
            return m;
        }

        public void SetRow(Matrix<T> v, int k)
        {
            for (int i = 0; i < Columns; i++)
                Data[k, i] = v[0, i];
        }


        /// <summary>
        /// Function returns the copy of this matrix
        /// </summary>
        public Matrix<T> Clone()
        {
            Matrix<T> matrix = new Matrix<T>(Rows, Columns);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    matrix[i, j] = Data[i, j];
            return matrix;
        }

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
        public static Matrix<T> Identity(int iRows, int iCols)
        {
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            var matrix = new Matrix<T>(iRows, iCols);

            for (int i = 0; i < Math.Min(iRows, iCols); i++)
                matrix[i, i] = Matrix<T>.One;
            return matrix;
        }

        public string ToString(string format, IFormatProvider formatProvider)
        {
            string s = "";
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++) s += String.Format("{0,5:0.00}", Data[i, j]) + " ";
                s += "\r\n";
            }
            return s;
        }

        public string ToString(string format)
        {
            return this.ToString(format, CultureInfo.CurrentCulture);
        }

        public override string ToString()                           // Function returns matrix as a string
        {
            return this.ToString("G", CultureInfo.CurrentCulture);
        }

        /// <summary>
        /// Matrix transpose, for any rectangular matrix
        /// </summary>
        public static Matrix<T> Transpose(Matrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            Matrix<T> t = new Matrix<T>(m.Columns, m.Rows);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Columns; j++)
                    t[j, i] = m[i, j];

            return t;
        }

        /// <summary>
        /// Matrix transpose, for any rectangular matrix
        /// </summary>
        public Matrix<T> Transpose()
        {
            return Matrix<T>.Transpose(this);
        }

        public Matrix<T> Invert()
        {
            return Matrix<T>.Invert(this);
        }

        public static Matrix<T> Invert(Matrix<T> m)
        {
            if (typeof(T) == typeof(float))
            {
                var t = m as Matrix<float>;
                return MatrixHelper.Invert(t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as Matrix<double>;
                return MatrixHelper.Invert(t) as Matrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static Matrix<T> Power(Matrix<T> m, int pow)           // Power matrix to exponent
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (pow == 0)
                return Identity(m.Rows, m.Columns);
            if (pow == 1)
                return m.Clone();
            if (pow == -1)
                return m.Invert();

            Matrix<T> x;
            if (pow < 0)
            {
                x = m.Invert();
                pow *= -1;
            }
            else x = m.Clone();

            Matrix<T> ret = Identity(m.Rows, m.Columns);
            while (pow != 0)
            {
                if ((pow & 1) == 1)
                    ret *= x;

                x *= x;
                pow >>= 1;
            }
            return ret;
        }

        public Matrix<T> Power(int pow)
        {
            return Matrix<T>.Power(this, pow);
        }

        /// <summary>
        /// Function returns permutation matrix "P" due to permutation vector "pi"
        /// </summary>
        public Matrix<T> Permutation()
        {
            var pi = decomposition.Value.Permutation;

            Matrix<T> matrix = Matrix<T>.Zeroes(Rows, Columns);
            for (int i = 0; i < Rows; i++)
                matrix[pi[i], i] = Matrix<T>.One;

            return matrix;
        }

        public bool Equals(Matrix<T> other)
        {
            if (other == null)
                return false;

            return Equals(this, other, Matrix<T>.Epsilon);
        }

        public static bool Equals(Matrix<T> m1, Matrix<T> m2)
        {
            return Equals(m1, m2, Matrix<T>.Epsilon);
        }
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;

                // Suitable nullity checks etc, of course :)
                hash = hash * 23 + this.Rows.GetHashCode();
                hash = hash * 23 + this.Columns.GetHashCode();
                hash = hash * 23 + this.Data.GetHashCode();
                return hash;
            }
        }

        public static bool Equals(Matrix<T> m1, Matrix<T> m2, T epsilon)
        {
            if (m1 == null || m2 == null)
                return false;

            if (m1.Columns != m2.Columns || m1.Rows != m2.Rows)
                return false;

            if (typeof(T) == typeof(int))
            {
                var t1 = m1 as Matrix<int>;
                var t2 = m2 as Matrix<int>;
                var e = (int)(object)epsilon;
                return MatrixHelper.Equals(t1, t2, e);
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = m1 as Matrix<float>;
                var t2 = m2 as Matrix<float>;
                var e = (float)(object)epsilon;
                return MatrixHelper.Equals(t1, t2, e);
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as Matrix<double>;
                var t2 = m2 as Matrix<double>;
                var e = (double)(object)epsilon;
                return MatrixHelper.Equals(t1, t2, e);
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        private static Decomposition<T> MakeLU(Matrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);
            Contract.Requires<InvalidOperationException>(m.IsSquare, "The matrix is not square!");

            if (typeof(T) == typeof(float))
            {
                var t = m as Matrix<float>;
                return MatrixHelper.LUDecomposition(t) as Decomposition<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as Matrix<double>;
                return MatrixHelper.LUDecomposition(t) as Decomposition<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public T Determinant()
        {
            return Determinant(this);
        }

        public static T Determinant(Matrix<T> m)
        {
            if (typeof(T) == typeof(float))
            {
                var t = m as Matrix<float>;
                return (T)(object)MatrixHelper.Determinant(t);
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as Matrix<double>;
                return (T)(object)MatrixHelper.Determinant(t);
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static Matrix<T> operator *(Matrix<T> m1, Matrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (typeof(T) == typeof(int))
            {
                var t1 = m1 as Matrix<int>;
                var t2 = m2 as Matrix<int>;
                return MatrixHelper.StrassenMultiply(t1, t2) as Matrix<T>;
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = m1 as Matrix<float>;
                var t2 = m2 as Matrix<float>;
                return MatrixHelper.StrassenMultiply(t1, t2) as Matrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as Matrix<double>;
                var t2 = m2 as Matrix<double>;
                return MatrixHelper.StrassenMultiply(t1, t2) as Matrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class."); 
        }

        public static Matrix<T> operator *(T c, Matrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var t = m as Matrix<int>;
                var c1 = (object)c;
                return MatrixHelper.Multiply((int)c1, t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as Matrix<float>;
                var c1 = (object)c;
                return MatrixHelper.Multiply((float)c1, t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as Matrix<double>;
                var c1 = (object)c;
                return MatrixHelper.Multiply((double)c1, t) as Matrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static Matrix<T> operator *(Matrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c * m;
        }

        public static Matrix<T> operator -(Matrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var t = m as Matrix<int>;
                return MatrixHelper.Negative(t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as Matrix<float>;
                return MatrixHelper.Negative(t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as Matrix<double>;
                return MatrixHelper.Negative(t) as Matrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }
        public static Matrix<T> operator +(Matrix<T> m1, Matrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (typeof(T) == typeof(int))
            {
                var t1 = m1 as Matrix<int>;
                var t2 = m2 as Matrix<int>;
                return MatrixHelper.Add(t1, t2) as Matrix<T>;
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = m1 as Matrix<float>;
                var t2 = m2 as Matrix<float>;
                return MatrixHelper.Add(t1, t2) as Matrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as Matrix<double>;
                var t2 = m2 as Matrix<double>;
                return MatrixHelper.Add(t1, t2) as Matrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class."); 
        }

        public static Matrix<T> operator -(Matrix<T> m1, Matrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (typeof(T) == typeof(int))
            {
                var t1 = m1 as Matrix<int>;
                var t2 = m2 as Matrix<int>;
                return MatrixHelper.Substract(t1, t2) as Matrix<T>;
            }
            else if (typeof(T) == typeof(float))
            {
                var t1 = m1 as Matrix<float>;
                var t2 = m2 as Matrix<float>;
                return MatrixHelper.Substract(t1, t2) as Matrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as Matrix<double>;
                var t2 = m2 as Matrix<double>;
                return MatrixHelper.Substract(t1, t2) as Matrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");  
        }

        public static Matrix<T> operator +(T c, Matrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var t = m as Matrix<int>;
                var c1 = (object)c;
                return MatrixHelper.Add((int)c1, t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as Matrix<float>;
                var c1 = (object)c;
                return MatrixHelper.Add((float)c1, t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as Matrix<double>;
                var c1 = (object)c;
                return MatrixHelper.Add((double)c1, t) as Matrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static Matrix<T> operator +(Matrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c + m;
        }

        public static Matrix<T> operator -(T c, Matrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                var t = m as Matrix<int>;
                var c1 = (object)c;
                return MatrixHelper.Substract((int)c1, t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as Matrix<float>;
                var c1 = (object)c;
                return MatrixHelper.Substract((float)c1, t) as Matrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as Matrix<double>;
                var c1 = (object)c;
                return MatrixHelper.Substract((double)c1, t) as Matrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static Matrix<T> operator -(Matrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c - m;
        }
    }
}
