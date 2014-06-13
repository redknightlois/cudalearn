using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class GpuMatrixTest
    {
        public GpuMatrixTest()
        {
            CudaLearnModule.Initialize();
            CudaLearnModule.AllowHandyForDebugButVerySlowGpuMemoryAccess = true;
        }

        [Fact]
        public void CreateGpuMatrixOfTWithNumericTypes()
        {
            var matrixInt = new GpuMatrix<int>(2, 2);
            var matrixDouble = new GpuMatrix<double>(2, 2);
            var matrixFloat = new GpuMatrix<float>(2, 2);
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
    }
}
