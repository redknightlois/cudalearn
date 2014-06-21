using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public partial class GpuMatrix<T> : IEquatable<GpuMatrix<T>>, IGpuMatrixStorage<T>, IDisposable where T : struct
    {
        public readonly int Rows;
        public readonly int Columns;

        /// <summary>
        /// Data is stored in Column-Major to be compliant with the BLAS packages
        /// </summary>
        private readonly CudaDeviceVariable<T> GpuData;

        public static readonly T Zero;
        public static readonly T One;
        public static readonly T NegativeOne;
        public static readonly T DefaultEpsilon;

        public static readonly Func<T,T> Negative = NegativeConstant;

        public static readonly T Epsilon;

        static GpuMatrix()
        {
            if (typeof(T) == typeof(int))
            {
                Zero = (T)(object)0;
                One = (T)(object)1;
                NegativeOne = (T)(object)-1;
                DefaultEpsilon = (T)(object)1;
            }
            else if (typeof(T) == typeof(float))
            {
                Zero = (T)(object)0f;
                One = (T)(object)1f;
                NegativeOne = (T)(object)-1f;
                DefaultEpsilon = (T)(object)0.00001f;
            }
            else if (typeof(T) == typeof(double))
            {
                Zero = (T)(object)0d;
                One = (T)(object)1d;
                NegativeOne = (T)(object)-1d;
                DefaultEpsilon = (T)(object)0.00001d;
            }
            else
            {
                Zero = default(T);
                One = default(T);
                NegativeOne = default(T);
                DefaultEpsilon = default(T);   
            }

            Epsilon = DefaultEpsilon;            
        }

        private static T NegativeConstant(T c)
        {            
            if (typeof(T) == typeof(int))
            {
                var aux = -(int)(object)c;
                return (T)(object)aux;
            }
            else if (typeof(T) == typeof(float))
            {
                var aux = -(float)(object)c;
                return (T)(object)aux;
            }
            else if (typeof(T) == typeof(double))
            {
                var aux = -(double)(object)c;
                return (T)(object)aux;
            }

            throw new NotSupportedException("Type: {0} is not supported by the GpuMatrix<T> class.");
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

            var aux = this;
            BlasMath.SetConstant(ref aux, value);
        }

        public GpuMatrix(GpuMatrix<T> matrix)
        {
            Contract.Requires<ArgumentNullException>( matrix != null );

            Rows = matrix.Rows;
            Columns = matrix.Columns;

            this.GpuData = new CudaDeviceVariable<T>( Rows * Columns );
            this.GpuData.CopyToDevice(matrix.GpuData);
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

                if (isLocked)
                    throw new InvalidOperationException("The matrix storage is in locked mode.");

                return GpuData[Rows * iCol + iRow];
            }
            set
            {
                Contract.Requires(iRow >= 0 && iRow < Rows);
                Contract.Requires(iCol >= 0 && iCol < Columns);

                if (isLocked)
                    throw new InvalidOperationException("The matrix storage is in locked mode.");

                if ( !CudaLearnModule.AllowHandyForDebugButVerySlowGpuMemoryAccess )
                    throw new InvalidOperationException("You cannot set values individually in the GpuMatrix<T> for performance reasons.");

                GpuData[Rows * iCol + iRow] = value;
            }
        }

        /// <summary>
        /// Function returns the copy of this matrix
        /// </summary>
        public GpuMatrix<T> Clone()
        {
            if (isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            return new GpuMatrix<T>(this);
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

        /// <summary>
        /// Matrix transpose, for any rectangular matrix
        /// </summary>
        public static GpuMatrix<T> Transpose(GpuMatrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (m.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

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
            if (m.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

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
            if (isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

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

            return BlasMath.Equals(m1, m2, epsilon);
        }

        public static GpuMatrix<T> operator *(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (m1.isLocked || m2.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            if (typeof(T) == typeof(float))
            {
                var t1 = m1 as GpuMatrix<float>;
                var t2 = m2 as GpuMatrix<float>;
                return BlasMath.Multiply(t1, t2) as GpuMatrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t1 = m1 as GpuMatrix<double>;
                var t2 = m2 as GpuMatrix<double>;
                return BlasMath.Multiply(t1, t2) as GpuMatrix<T>;
            }

            throw new NotSupportedException("Type: {0} is not supported by the GpuMatrix<T> class.");
        }

        public static GpuMatrix<T> operator *(T c, GpuMatrix<T> m)
        {
            Contract.Requires(typeof(T) == typeof(int) || typeof(T) == typeof(float) || typeof(T) == typeof(double));
            Contract.Requires<ArgumentNullException>(m != null);

            if (m.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            return BlasMath.Axpb(m, c, GpuMatrix<T>.Zero);
        }

        public static GpuMatrix<T> operator *(GpuMatrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c * m;
        }

        public static GpuMatrix<T> operator -(GpuMatrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);
            
            if (m.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            var negative = GpuMatrix<T>.Zeroes(m.Rows, m.Columns);
            BlasMath.AxpyInPlace<T>(m, GpuMatrix<T>.NegativeOne, ref negative);
            return negative;
        }
        public static GpuMatrix<T> operator +(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<NotSupportedException>(typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int));
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (m1.isLocked || m2.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            return BlasMath.Axpy(m1, GpuMatrix<T>.One, m2) as GpuMatrix<T>;
        }

        public static GpuMatrix<T> operator -(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<NotSupportedException>(typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int));
            Contract.Requires<ArgumentNullException>(m1 != null && m2 != null);

            if (m1.isLocked || m2.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            return BlasMath.Axpy(m2, GpuMatrix<T>.NegativeOne, m1);
        }

        public static GpuMatrix<T> operator +(T c, GpuMatrix<T> m)
        {
            Contract.Requires(typeof(T) == typeof(int) || typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(int));
            Contract.Requires<ArgumentNullException>(m != null);
            
            if (m.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            return BlasMath.Axpb(m, GpuMatrix<T>.One, c);
        }

        public static GpuMatrix<T> operator +(GpuMatrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c + m;
        }

        public static GpuMatrix<T> operator -(T c, GpuMatrix<T> m)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (m.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            return BlasMath.Axpb(m, GpuMatrix<T>.One, NegativeConstant(c)) as GpuMatrix<T>;            
        }

        public static GpuMatrix<T> operator -(GpuMatrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            return c - m;
        }

        public static GpuMatrix<T> operator /(GpuMatrix<T> m, T c)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            if (typeof(T) == typeof(int))
            {
                throw new NotImplementedException();
            }
            else if (typeof(T) == typeof(float))
            {
                var t = m as GpuMatrix<float>;
                float c1 = (float)(object)c;
                return t * (1.0f / c1) as GpuMatrix<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var t = m as GpuMatrix<double>;
                double c1 = (double)(object)c;
                return t * (1.0f / c1) as GpuMatrix<T>;
            }
            throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static GpuMatrix<T> operator /(T c, GpuMatrix<T> m)
        {
            return m / c;
        }

        public static GpuMatrix<T> operator ^(GpuMatrix<T> m, int x)
        {
            Contract.Requires<ArgumentNullException>(m != null);

            throw new NotImplementedException();
        }

        public static GpuMatrix<T> operator >(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);

            throw new NotImplementedException();
        }

        public static GpuMatrix<T> operator >=(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);

            throw new NotImplementedException();
        }

        public static GpuMatrix<T> operator <(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);

            throw new NotImplementedException();
        }

        public static GpuMatrix<T> operator <=(GpuMatrix<T> m1, GpuMatrix<T> m2)
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);

            throw new NotImplementedException();
        }

        CudaDeviceVariable<T> IGpuMatrixStorage<T>.GetDeviceMemory()
        {
            return GpuData;
        }


        public string ToString(string format, IFormatProvider formatProvider)
        {
            if (isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            if (Rows * Columns > 36)
                return string.Format( "M[{0},{1}] is too large to convert to string. Use a dedicated dumper.", Rows, Columns );

            string s = "";
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++) s += String.Format("{0,5:0.00}", this[i, j]) + " ";
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

        public static implicit operator GpuMatrix<T>(Matrix<T> m)
        {
            var r = new GpuMatrix<T>(m.Rows, m.Columns);

            var mt = ((IHostMatrixStorage<T>)m).GetHostMemory();
            var rt = ((IGpuMatrixStorage<T>)r).GetDeviceMemory();

            rt.CopyToDevice(mt);

            return r;
        }

        public static explicit operator Matrix<T> ( GpuMatrix<T> m )
        {
            if (m.isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            var r = new Matrix<T>(m.Rows, m.Columns);

            var mt = ((IGpuMatrixStorage<T>)m).GetDeviceMemory();            
            var rt = ((IHostMatrixStorage<T>)r).GetHostMemory();

            mt.CopyToHost(rt);
            
            return r;
        }

        public virtual void Dispose()
        {
            if (isLocked)
                throw new InvalidOperationException("The matrix storage is in locked mode.");

            if (this.GpuData != null)
                this.GpuData.Dispose();
        }

        [ContractInvariantMethod]
        private void ObjectInvariants()
        {
            Contract.Invariant(this.Rows > 0);
            Contract.Invariant(this.Columns > 0);
            Contract.Invariant(this.GpuData != null);
            Contract.Invariant(this.GpuData.Size == this.Rows * this.Columns);
        }


        private bool isLocked = false;

        void IGpuMatrixStorage<T>.Lock()
        {
            this.isLocked = true;
        }

        void IGpuMatrixStorage<T>.Unlock()
        {
            this.isLocked = false;
        }
    }
}
