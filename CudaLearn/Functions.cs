using MathNet.Numerics.LinearAlgebra;
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
        public static Matrix<T> AddVectorOnEachColumn<T>(this Matrix<T> m1, Vector<T> v, T scale) where T : struct, IEquatable<T>, IFormattable
        {
            Contract.Requires(m1 != null);
            Contract.Requires(v != null);

            if (m1 is Matrix<float>)
            {
                var fm1 = m1 as Matrix<float>;
                var fv = v as Vector<float>;
                var fscale = (float)(object)scale;

                return fm1.MapIndexed((i, j, value) => fm1.At(i, j) + fscale * fv.At(j)) as Matrix<T>;
            }
            else if (m1 is Matrix<double>)
            {
                var dm1 = m1 as Matrix<double>;
                var dv = v as Vector<double>;
                var dscale = (double)(object)scale;

                return dm1.MapIndexed((i, j, value) => dm1.At(i, j) + dscale * dv.At(j)) as Matrix<T>;
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static Matrix<T> AddVectorOnEachColumn<T>(this Matrix<T> m1, Vector<T> v) where T : struct, IEquatable<T>, IFormattable
        {
            Contract.Requires(m1 != null);
            Contract.Requires(v != null);

            return AddVectorOnEachColumn(m1, v, Matrix<T>.One);
        }

        public static Matrix<T> AddVectorOnEachRow<T>(this Matrix<T> m1, Vector<T> v, T scale) where T : struct, IEquatable<T>, IFormattable
        {
            Contract.Requires(m1 != null);
            Contract.Requires(v != null);

            if (m1 is Matrix<float>)
            {
                var fm1 = m1 as Matrix<float>;
                var fv = v as Vector<float>;
                var fscale = (float)(object)scale;

                return fm1.MapIndexed((i, j, value) => fm1.At(i, j) + fscale * fv.At(i)) as Matrix<T>;
            }
            else if (m1 is Matrix<double>)
            {
                var dm1 = m1 as Matrix<double>;
                var dv = v as Vector<double>;
                var dscale = (double)(object)scale;

                return dm1.MapIndexed((i, j, value) => dm1.At(i, j) + dscale * dv.At(i)) as Matrix<T>;
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }

        public static Matrix<T> AddVectorOnEachRow<T>(this Matrix<T> m1, Vector<T> v) where T : struct, IEquatable<T>, IFormattable
        {
            Contract.Requires(m1 != null);
            Contract.Requires(v != null);

            return AddVectorOnEachRow(m1, v, Matrix<T>.One);
        }

        public static Matrix<T> BinaryLess<T>(this Matrix<T> m1, Matrix<T> m2) where T : struct, IEquatable<T>, IFormattable
        {
            Contract.Requires(m1 != null);
            Contract.Requires(m2 != null);

            if (m1 is Matrix<float>)
            {
                var fm1 = m1 as Matrix<float>;
                var fm2 = m2 as Matrix<float>;

                return fm1.MapIndexed((i, j, v) => v < fm2.At(i, j) ? Matrix<float>.One : Matrix<float>.Zero) as Matrix<T>;
            }
            else if (m1 is Matrix<double>)
            {
                var dm1 = m1 as Matrix<double>;
                var dm2 = m2 as Matrix<double>;

                return dm1.MapIndexed((i, j, v) => v < dm2.At(i, j) ? Matrix<double>.One : Matrix<double>.Zero) as Matrix<T>;
            }
            else throw new NotSupportedException("Type: {0} is not supported by the Matrix<T> class.");
        }
    }
}
