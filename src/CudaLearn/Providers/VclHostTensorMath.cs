using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CudaLearn.Providers.Impl;

namespace CudaLearn.Providers
{
    public class VclHostTensorMath : ICpuTensorMath
    {
        public void Gemm(BlasTranspose transA, BlasTranspose transB, int m, int n, int k, double alpha, ArraySlice<double> a, ArraySlice<double> b, double beta, ArraySlice<double> c)
        {
            VclNativeMethods.vclGemm(transA, transB, m, n, k, alpha, a.Array, a.Offset, b.Array, b.Offset, beta, c.Array, c.Offset);
        }

        public void Gemv(BlasTranspose transA, int m, int n, double alpha, ArraySlice<double> a, ArraySlice<double> x, double beta, ArraySlice<double> y)
        {
            VclNativeMethods.vclGemv(transA, m, n, alpha, a.Array, a.Offset, a.Length, x.Array, x.Offset, x.Length, beta, y.Array, y.Offset, y.Length);
        }

        public void Axpy(double alpha, ArraySlice<double> x, ArraySlice<double> y)
        {
            if (x.Length != y.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclAxpy(x.Length, alpha, x.Array, x.Offset, y.Array, y.Offset);
        }

        public void Axpby(double alpha, ArraySlice<double> x, double beta, ArraySlice<double> y)
        {
            if (x.Length != y.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclAxpby(x.Length, alpha, x.Array, x.Offset, beta, y.Array, y.Offset);
        }

        public void Set(double alpha, ArraySlice<double> y)
        {
            VclNativeMethods.vclSet(y.Length, alpha, y.Array, y.Offset);
        }

        public void Set(int alpha, ArraySlice<double> y)
        {
            Set((double)alpha, y);
        }

        public void Copy(ArraySlice<double> x, ArraySlice<double> y)
        {
            if (x.Length != y.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclCopy(x.Length, x.Array, x.Offset, y.Array, y.Offset);
        }

        public void Add(double alpha, ArraySlice<double> y)
        {
            VclNativeMethods.vclAddScalar(y.Length, alpha, y.Array, y.Offset);
        }

        public void Add(ArraySlice<double> a, ArraySlice<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclAdd(a.Length, a.Array, a.Offset, b.Array, b.Offset);
        }

        public void Substract(ArraySlice<double> a, ArraySlice<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclSubstract(a.Length, a.Array, a.Offset, b.Array, b.Offset);
        }

        public void Multiply(ArraySlice<double> a, ArraySlice<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclMultiply(a.Length, a.Array, a.Offset, b.Array, b.Offset);
        }

        public void Divide(ArraySlice<double> a, ArraySlice<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclDivide(a.Length, a.Array, a.Offset, b.Array, b.Offset);
        }

        public void Powx(ArraySlice<double> a, ArraySlice<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclPowx(a.Length, a.Array, a.Offset, b.Array, b.Offset);
        }

        public void Square(ArraySlice<double> a, ArraySlice<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclSquare(a.Length, a.Array, a.Offset, b.Array, b.Offset);
        }

        public void Exp(ArraySlice<double> a, ArraySlice<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclExp(a.Length, a.Array, a.Offset, b.Array, b.Offset);
        }

        public void Abs(ArraySlice<double> a, ArraySlice<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclAbs(a.Length, a.Array, a.Offset, b.Array, b.Offset);
        }

        public double Dot(ArraySlice<double> x, ArraySlice<double> y)
        {
            if (x.Length != y.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            return VclNativeMethods.vclDot(x.Length, x.Array, x.Offset, y.Array, y.Offset);
        }

        public double Dot(ArraySlice<double> x, int incx, ArraySlice<double> y, int incy)
        {
            if (incx <= 0 || incy <= 0)
                throw new ArgumentException(Constants.Exceptions.IncrementsMustBePositive);

            return VclNativeMethods.vclDotEx(x.Length, x.Array, x.Offset, incx, y.Array, y.Offset, incy);
        }

        public double Asum(ArraySlice<double> x)
        {
            return VclNativeMethods.vclAsum(x.Length, x.Array, x.Offset);
        }

        public void Scale(double alpha, ArraySlice<double> x)
        {
            VclNativeMethods.vclScaleInline(x.Length, alpha, x.Array, x.Offset);
        }

        public void Scale(double alpha, ArraySlice<double> x, ArraySlice<double> y)
        {
            if (x.Length != y.Length)
                throw new ArgumentException(Constants.Exceptions.ArrayLengthMustBeEqual);

            VclNativeMethods.vclScale(x.Length, alpha, x.Array, x.Offset, y.Array, y.Offset);
        }

        public int GetVersion()
        {
            return VclNativeMethods.vclGetVersion();
        }
    }
}
