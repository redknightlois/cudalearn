using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{
    public class GpuMatrixTest : MathTestsBase
    {
        public GpuMatrixTest()
        {
            CudaLearnModule.Initialize();
            CudaLearnModule.AllowHandyForDebugButVerySlowGpuMemoryAccess = true;
        }

        [Theory]
        [InlineData(1)]
        [InlineData(1.0d)]
        [InlineData(1.0f)]
        public void CreateGpuMatrixOfTWithNumericTypes<T>(T dummy) where T : struct
        {
            var matrix = new GpuMatrix<T>(2,2, dummy);
        }

        private struct SampleStruct
        {
        }

        [Fact]
        public void CreateGenericMatrixWithInvalidTypes()
        {            
            Assert.Throws<NotSupportedException>(() => new GpuMatrix<byte>(2, 2));
            Assert.Throws<NotSupportedException>(() => new GpuMatrix<SampleStruct>(2, 2));
            Assert.Throws<NotSupportedException>(() => new GpuMatrix<sbyte>(2, 2));
        }

        [Theory]
        [InlineData(10)]
        [InlineData(10.0d)]
        [InlineData(10.0f)]
        public void ConstructorsShouldWork<T>(T value) where T : struct
        {
            var m1 = new GpuMatrix<T>(2, 2, value);
            var m2 = GpuMatrix<T>.Zeroes(2, 2);
            var m3 = GpuMatrix<T>.Identity(2, 2);

            for (int i = 0; i < m1.Rows; i++)
            {
                for (int j = 0; j < m1.Columns; j++)
                {
                    Assert.Equal(value, m1[i, j]);
                    Assert.Equal(GpuMatrix<T>.Zero, m2[i, j]);
                    if (i == j)
                        Assert.Equal(GpuMatrix<T>.One, m3[i, j]);
                    else
                        Assert.Equal(GpuMatrix<T>.Zero, m3[i, j]);
                }
            }
        }

        [Fact]
        public void MultiplyWithNonSquareMatrix()
        {
            var m1 = new GpuMatrix<float>(2, 3);
            var m2 = new GpuMatrix<float>(3, 2);

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

            var expected = new GpuMatrix<float>(2, 2);
            expected[0, 0] = 7;
            expected[0, 1] = 8;
            expected[1, 0] = 10;
            expected[1, 1] = 11;

            Assert.Equal(expected, result);
        }

        [Fact]
        public void MultiplyWithSquareMatrix()
        {
            var m1 = new GpuMatrix<float>(2, 2);
            var m2 = new GpuMatrix<float>(2, 2);

            m1[0, 0] = 1;
            m1[0, 1] = 1;
            m1[1, 0] = 1;
            m1[1, 1] = 1;

            m2[0, 0] = 1;
            m2[0, 1] = 2;
            m2[1, 0] = 3;
            m2[1, 1] = 4;

            var result = m1 * m2;

            var expected = new GpuMatrix<float>(2, 2);
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
        public void MultiplyIdentityWithSquareMatrix()
        {
            var m1 = new GpuMatrix<float>(2, 2);
            var m2 = new GpuMatrix<float>(2, 2);

            m1[0, 0] = 1;
            m1[0, 1] = 1;
            m1[1, 0] = 1;
            m1[1, 1] = 1;

            m2[0, 0] = 1;
            m2[0, 1] = 2;
            m2[1, 0] = 3;
            m2[1, 1] = 4;

            var result = m1 * m2;

            var expected = new GpuMatrix<float>(2, 2);
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
        public void CloneOnDeviceWithInt()
        {
            var m1 = new GpuMatrix<int>(2, 2);
            m1[0, 0] = 1;
            m1[0, 1] = 5;
            m1[1, 0] = 3;
            m1[1, 1] = 2;

            var m2 = new GpuMatrix<int>(m1);
            var m3 = m1.Clone();

            Assert.Equal(m1, m2);
            Assert.Equal(m2, m3);

            Assert.Equal(1, m3[0, 0]);
            Assert.Equal(5, m3[0, 1]);
            Assert.Equal(3, m3[1, 0]);
            Assert.Equal(2, m3[1, 1]);
        }

        [Fact]
        public void CloneOnDeviceWithFloat()
        {
            var m1 = new GpuMatrix<float>(2, 2);
            m1[0, 0] = 1;
            m1[0, 1] = 5;
            m1[1, 0] = 3;
            m1[1, 1] = 2;

            var m2 = new GpuMatrix<float>(m1);
            var m3 = m1.Clone();

            Assert.Equal(m1, m2);
            Assert.Equal(m2, m3);

            Assert.Equal(1, m3[0, 0]);
            Assert.Equal(5, m3[0, 1]);
            Assert.Equal(3, m3[1, 0]);
            Assert.Equal(2, m3[1, 1]);
        }


        [Fact]
        public void AdditionAndSubstractionWithFloatMatrix()
        {
            var m1 = new GpuMatrix<float>(2, 2);
            m1[0, 0] = 1;
            m1[0, 1] = 5;
            m1[1, 0] = 3;
            m1[1, 1] = 2;

            var m2 = -m1;
            var m3 = m1.Clone();

            var result = m1 + m2;
            Assert.Equal(GpuMatrix<float>.Zeroes(2, 2), result);

            result = m1 - m3;
            Assert.Equal(GpuMatrix<float>.Zeroes(2, 2), result);

            result = m1 - 5;

            var e1 = new GpuMatrix<float>(2, 2);
            e1[0, 0] = -4;
            e1[0, 1] = 0;
            e1[1, 0] = -2;
            e1[1, 1] = -3;
            Assert.Equal(e1, result);

            result = result + 5;
            Assert.Equal(m1, result);
        }

        [Fact]
        public void AdditionAndSubstractionWithIntMatrix()
        {
            var m1 = new GpuMatrix<int>(2, 2);
            m1[0, 0] = 1;
            m1[0, 1] = 5;
            m1[1, 0] = 3;
            m1[1, 1] = 2;

            var m2 = -m1;
            var m3 = m1.Clone();

            var result = m1 + m2;
            Assert.Equal(GpuMatrix<int>.Zeroes(2, 2), result);

            result = m1 - m3;
            Assert.Equal(GpuMatrix<int>.Zeroes(2, 2), result);

            result = m1 - 5;

            var e1 = new GpuMatrix<int>(2, 2);
            e1[0, 0] = -4;
            e1[0, 1] = 0;
            e1[1, 0] = -2;
            e1[1, 1] = -3;
            Assert.Equal(e1, result);

            result = result + 5;
            Assert.Equal(m1, result);
        }

        [Fact]
        public void AdditionAndSubstractionWithDoubleMatrix()
        {
            var m1 = new GpuMatrix<double>(2, 2);
            m1[0, 0] = 1;
            m1[0, 1] = 5;
            m1[1, 0] = 3;
            m1[1, 1] = 2;

            var m2 = -m1;
            var m3 = m1.Clone();

            var result = m1 + m2;
            Assert.Equal(GpuMatrix<double>.Zeroes(2, 2), result);

            result = m1 - m3;
            Assert.Equal(GpuMatrix<double>.Zeroes(2, 2), result);

            result = m1 - 5;

            var e1 = new GpuMatrix<double>(2, 2);
            e1[0, 0] = -4;
            e1[0, 1] = 0;
            e1[1, 0] = -2;
            e1[1, 1] = -3;
            Assert.Equal(e1, result);

            result = result + 5;
            Assert.Equal(m1, result);
        }

    }
}
