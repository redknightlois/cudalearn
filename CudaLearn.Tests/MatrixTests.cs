using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class MatrixTest : MathTestsBase
    {
        [Fact]
        public void CreateGenericMatrixWithNumericTypes()
        {
            var matrixInt = new Matrix<int>(2, 2);
            var matrixDouble = new Matrix<double>(2, 2);
            var matrixFloat = new Matrix<float>(2, 2);
        }

        private struct SampleStruct
        {
        }

        [Fact]
        public void CreateGenericMatrixWithInvalidTypes()
        {
            Assert.Throws<NotSupportedException>(() => new Matrix<byte>(2, 2));
            Assert.Throws<NotSupportedException>(() => new Matrix<SampleStruct>(2, 2));
            Assert.Throws<NotSupportedException>(() => new Matrix<sbyte>(2, 2));
        }

        [Fact]
        public void MultiplyWithIdentityMatrix()
        {
            var m1 = new Matrix<int>(2, 2);
            var m2 = new Matrix<int>(2, 2);

            m1[0, 0] = 1;
            m1[1, 1] = 1;

            m2[0, 0] = 1;
            m2[1, 0] = 2;
            m2[0, 1] = 3;
            m2[1, 1] = 4;

            var result = m1 * m2;

            Assert.Equal(m2, result);
        }

        [Fact]
        public void GetColumnsAndRowsOfMatrix()
        {
            var m = new Matrix<float>(2, 2);
            m[0, 0] = 1;
            m[0, 1] = 3;
            m[1, 0] = 2;
            m[1, 1] = 4;

            var c = m.GetColumn(0);
            Assert.Equal(1, c.Columns);
            Assert.Equal(2, c.Rows);
            Assert.Equal(1, c[0, 0]);
            Assert.Equal(2, c[1, 0]);

            var r = m.GetRow(0);
            Assert.Equal(2, r.Columns);
            Assert.Equal(1, r.Rows);
            Assert.Equal(1, r[0, 0]);
            Assert.Equal(3, r[0, 1]);
        }

        [Fact]
        public void SetColumnsAndRowsOfMatrix()
        {
            var m = new Matrix<float>(2, 2);
            m[0, 0] = 1;
            m[0, 1] = 3;
            m[1, 0] = 2;
            m[1, 1] = 4;

            var c0 = new Matrix<float>(2, 1);
            c0[0, 0] = 1;
            c0[1, 0] = 2;

            var c1 = new Matrix<float>(2, 1);
            c1[0, 0] = 3;
            c1[1, 0] = 4;

            var c = Matrix<float>.Zeroes(2, 2);
            c.SetColumn(c0, 0);
            c.SetColumn(c1, 1);

            Assert.Equal(m, c);

            var r0 = new Matrix<float>(1, 2);
            r0[0, 0] = 1;
            r0[0, 1] = 3;

            var r1 = new Matrix<float>(1, 2);
            r1[0, 0] = 2;
            r1[0, 1] = 4;

            var r = Matrix<float>.Zeroes(2, 2);
            r.SetRow(r0, 0);
            r.SetRow(r1, 1);

            Assert.Equal(m, r);
        }

        [Fact]
        public void MultiplyWithSquareMatrix()
        {
            var m1 = new Matrix<float>(2, 2);
            var m2 = new Matrix<float>(2, 2);

            m1[0, 0] = 1;
            m1[0, 1] = 1;
            m1[1, 0] = 1;
            m1[1, 1] = 1;

            m2[0, 0] = 1;
            m2[0, 1] = 2;
            m2[1, 0] = 3;
            m2[1, 1] = 4;

            var result = m1 * m2;

            var expected = new Matrix<float>(2, 2);
            expected[0, 0] = 4;
            expected[0, 1] = 6;
            expected[1, 0] = 4;
            expected[1, 1] = 6;

            Assert.Equal(expected, result);

            result = m2 * m1;

            expected[0, 0] = 3;
            expected[0, 1] = 3;
            expected[1, 0] = 7;
            expected[1, 1] = 7;

            Assert.Equal(expected, result);
        }

        [Fact]
        public void AdditionAndSubstractionWithMatrix()
        {
            var m1 = new Matrix<int>(2, 2);
            m1[0, 0] = 1;
            m1[0, 1] = 5;
            m1[1, 0] = 3;
            m1[1, 1] = 2;

            var m2 = -m1;
            var m3 = m1.Clone();

            var result = m1 + m2;
            Assert.Equal(Matrix<int>.Zeroes(2, 2), result);

            result = m1 - m3;
            Assert.Equal(Matrix<int>.Zeroes(2, 2), result);

            result = m1 - 5;

            var e1 = new Matrix<int>(2, 2);            
            e1[0, 0] = -4;
            e1[0, 1] = 0;
            e1[1, 0] = -2;
            e1[1, 1] = -3;

            Assert.Equal(e1, result);

            result = result + 5;
            Assert.Equal(m1, result);
        }

        [Fact]
        public void MultiplyWithNonSquareMatrix()
        {
            var m1 = new Matrix<float>(2, 3);
            var m2 = new Matrix<float>(3, 2);

            m1[0, 0] = 1;
            m1[0, 1] = 2;
            m1[0, 2] = 3;
            m1[1, 0] = 2;
            m1[1, 1] = 3;
            m1[1, 2] = 4;

            m2[0, 0] = 1;
            m2[0, 1] = 0;
            m2[1, 0] = 0;
            m2[1, 1] = 1;
            m2[2, 0] = 2;
            m2[2, 1] = 2;

            var result = m1 * m2;

            var expected = new Matrix<float>(2, 2);
            expected[0, 0] = 7;
            expected[0, 1] = 8;
            expected[1, 0] = 10;
            expected[1, 1] = 11;

            Assert.Equal(expected, result);
        }

        [Fact]
        public void DeterminantFromIntMatrixIsNotSupported()
        {
            var i = Matrix<int>.Identity(2);
            Assert.Throws<NotSupportedException>(() => i.Determinant());
        }


        [Fact]
        public void DeterminantFromMatrix()
        {
            var m1 = new Matrix<float>(2, 2);
            var m2 = new Matrix<float>(2, 2);

            m1[0, 0] = 1;
            m1[0, 1] = 1;
            m1[1, 0] = 1;
            m1[1, 1] = 1;

            m2[0, 0] = 1;
            m2[0, 1] = 2;
            m2[1, 0] = 3;
            m2[1, 1] = 4;

            Assert.True(EqualsWithEpsilon(0, m1.Determinant()));
            Assert.True(EqualsWithEpsilon(-2, m2.Determinant()));
        }


        [Fact]
        public void InverseOfMatrix()
        {
            var m = new Matrix<float>(2, 2);

            m[0, 0] = 1;
            m[0, 1] = 2;
            m[1, 0] = 3;
            m[1, 1] = 4;

            var mInverse = new Matrix<float>(2, 2);
            mInverse[0, 0] = -2f;
            mInverse[0, 1] = 1f;
            mInverse[1, 0] = 1.5f;
            mInverse[1, 1] = -0.5f;

            Assert.Equal(mInverse, m.Invert());

            Assert.Equal(Matrix<float>.Identity(2), m * mInverse);
            Assert.Equal(Matrix<float>.Identity(2), mInverse * m);
        }

        [Fact]
        public void TransposeWithNonSquareMatrix()
        {
            var m = new Matrix<int>(2, 3);
            m[0, 0] = 1;
            m[0, 1] = 2;
            m[0, 2] = 3;
            m[1, 0] = 2;
            m[1, 1] = 3;
            m[1, 2] = 4;

            var t = m.Transpose();

            var expected = new Matrix<int>(3, 2);
            expected[0, 0] = 1;
            expected[1, 0] = 2;
            expected[2, 0] = 3;
            expected[0, 1] = 2;
            expected[1, 1] = 3;
            expected[2, 1] = 4;

            Assert.Equal(expected, t);
            Assert.Equal(m, t.Transpose());

        }

        [Fact]
        public void MultiplyHugeMatrixWithIdentity()
        {
            var m = new Matrix<float>(256, 256);

            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Columns; j++)
                    m[i, j] = 1;

            var identity = Matrix<float>.Identity(256);

            var m1 = m * identity;
            var m2 = identity * m;

            Assert.Equal(m, m1);
            Assert.Equal(m, m2);
        }

        [Fact]
        public void MultiplyHugeMatrixWithScalar()
        {
            var m = new Matrix<int>(256, 512);
            var expected = new Matrix<int>(256, 512);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Columns; j++)
                {
                    m[i, j] = 1;
                    expected[i, j] = 7;
                }

            var result = 7 * m;

            Assert.Equal(expected, result);
        }

        [Fact]
        public void InverseHugeMatrixWithIdentity()
        {
            var generator = new Random(123);

            var m = new Matrix<double>(256, 256);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Columns; j++)
                    m[i, j] = generator.NextDouble();            

            var mInverse = m.Invert();

            var identity = Matrix<double>.Identity(256);

            Assert.Equal(identity, m * mInverse);
            Assert.Equal(identity, mInverse * m);
        }

        [Fact]
        public void SumOverAxis()
        {
            var m1 = new Matrix<float>(2, 3);

            // 1 2 3
            // 2 3 4            
            m1[0, 0] = 1;
            m1[0, 1] = 2;
            m1[0, 2] = 3;
            m1[1, 0] = 4;
            m1[1, 1] = 5;
            m1[1, 2] = 6;

            var sumCols = Functions.Sum(m1, Axis.Columns);
            Assert.Equal(1, sumCols.Rows);
            Assert.Equal(3, sumCols.Columns);

            var gpuSumCols = Functions.Sum((GpuMatrix<float>)m1, Axis.Columns);
            Assert.Equal(1, gpuSumCols.Rows);
            Assert.Equal(3, gpuSumCols.Columns);

            var resultColumnSum = new Matrix<float>(1, 3);
            resultColumnSum[0, 0] = 5;
            resultColumnSum[0, 1] = 7;
            resultColumnSum[0, 2] = 9;
            Assert.Equal(resultColumnSum, sumCols);
            Assert.Equal(resultColumnSum, (Matrix<float>)gpuSumCols);

            var sumRows = Functions.Sum(m1, Axis.Rows);
            Assert.Equal(2, sumRows.Rows);
            Assert.Equal(1, sumRows.Columns);

            var gpuSumRows = Functions.Sum((GpuMatrix<float>)m1, Axis.Rows);
            Assert.Equal(2, gpuSumRows.Rows);
            Assert.Equal(1, gpuSumRows.Columns);

            var resultRowSum = new Matrix<float>(2, 1);
            resultRowSum[0, 0] = 6;
            resultRowSum[1, 0] = 15;
            Assert.Equal(resultRowSum, sumRows);
            Assert.Equal(resultRowSum, (Matrix<float>)gpuSumRows);
        }

        [Fact]
        public void SumVectorOverAxis()
        {
            var m1 = new Matrix<float>(2, 3);

            // 1 0 1
            // 0 1 0            
            m1[0, 0] = 1;
            m1[0, 1] = 0;
            m1[0, 2] = 1;
            m1[1, 0] = 0;
            m1[1, 1] = 1;
            m1[1, 2] = 0;

            var gm1 = (GpuMatrix<float>)m1;

            var result = new Matrix<float>(2, 3);
            result[0, 0] = 2;
            result[0, 1] = 1;
            result[0, 2] = 2;
            result[1, 0] = 1;
            result[1, 1] = 2;
            result[1, 2] = 1;

            var columnVector = new Matrix<float>(1, 3, 1);
            var sumCols = m1 + columnVector;
            var gpuSumCols = gm1 + columnVector;

            var rowsVector = new Matrix<float>(2, 1, 1);
            var sumRows = m1 + rowsVector;
            var gpuSumRows = gm1 + rowsVector;

            Assert.Equal(result, sumCols);
            Assert.Equal(result, gpuSumCols);

            Assert.Equal(result, sumRows);
            Assert.Equal(result, gpuSumRows);
        }


        [Fact]
        public void SumVectorOverAxisBig()
        {
            var m1 = Matrix<float>.Uniform(512, 2400);
            var cv = Matrix<float>.Uniform(1, 2400);
            var rv = Matrix<float>.Uniform(512, 1);

            var cResult = m1 + cv;
            var rResult = m1 + rv;

            var gm1 = (Matrix<float>)m1;
            var gcv = (Matrix<float>)cv;
            var grv = (Matrix<float>)rv;

            var gcResult = gm1 + gcv;
            var grResult = gm1 + grv;

            Assert.Equal(cResult, gcResult);
            Assert.Equal(rResult, grResult);
        }
    }
}
