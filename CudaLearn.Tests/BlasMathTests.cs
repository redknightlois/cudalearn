using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class BlasMathTests : MathTestsBase
    {
        [Fact]
        public void AbsoluteSumSuccessful()
        {
            CudaLearnModule.AllowHandyForDebugButVerySlowGpuMemoryAccess = true;

            var generator = new Random(123);
            int columns = 450;

            var m = new GpuMatrix<double>(1, columns);
            var mc = new Matrix<double>(1, columns);
            for (int j = 0; j < m.Columns; j++)
            {
                double aux = generator.Next(150) / 150.0f;
                m[0, j] = aux;
                mc[0, j] = aux;
            }

            double gpuSum = BlasMath.SumMagnitudes(m);

            double cpuSum = 0;
            for (int j = 0; j < columns; j++)
            {
                double aux = mc[0, j];
                cpuSum += aux >= 0 ? aux : -aux;
            }

            Assert.InRange(gpuSum - cpuSum, -Matrix<double>.DefaultEpsilon, Matrix<double>.DefaultEpsilon);
        }

        [Fact]
        public void GemmWithSquared()
        {
            Matrix<float> x = new Matrix<float>(4, 4, 2);
            var y = Matrix<float>.Identity(4);
            var z = new Matrix<float>(4, 4, 1);

            var op = BlasMath.Gemm(2.0f, x, y, 2.0f, z, BlasOperation.NonTranspose, BlasOperation.NonTranspose);

            var gemm = 2.0f * x * y + (2.0f * z);
            Assert.Equal(gemm, op);
        }

        [Fact]
        public void GemmWithNonSquared()
        {
            var x = new Matrix<double>(4, 2, 2);
            var y = new Matrix<double>(2, 3, 4);
            var z = new Matrix<double>(4, 3, 1);

            var op = BlasMath.Gemm(2.0d, x, y, 2.0d, z, BlasOperation.NonTranspose, BlasOperation.NonTranspose);

            var gemm = 2.0d * x * y + (2.0d * z);
            Assert.Equal(gemm, op);
        }

        [Fact]
        public void GemmWithNonSquaredAllTransposed()
        {
            var x = new Matrix<double>(2, 4, 2);
            var y = new Matrix<double>(3, 2, 4);
            var z = new Matrix<double>(4, 3, 1);

            var gemm = 2.0d * x.Transpose() * y.Transpose() + (2.0d * z);

            var op = BlasMath.Gemm(2.0d, x, y, 2.0d, z, BlasOperation.Transpose, BlasOperation.Transpose);
            Assert.Equal(gemm, op);
        }

        [Fact]
        public void GemmWithNonSquaredFirstTransposed()
        {
            var x = new Matrix<float>(2, 4, 2);
            var y = new Matrix<float>(2, 3, 4);
            var z = new Matrix<float>(4, 3, 1);

            var gemm = 2.0f * x.Transpose() * y + (2.0f * z);

            var op = BlasMath.Gemm(2.0f, x, y, 2.0f, z, BlasOperation.Transpose, BlasOperation.NonTranspose);
            
            Assert.Equal(gemm, op);
        }

        [Fact]
        public void GemmWithNonSquaredSecondTransposed()
        {
            var x = new Matrix<float>(3, 2, 2);
            var y = new Matrix<float>(3, 2, 4);
            var z = new Matrix<float>(3, 3, 1);

            var gemm = 1.0f * x * y.Transpose() + (2.0f * z);

            var op = BlasMath.Gemm(1.0f, x, y, 2.0f, z, BlasOperation.NonTranspose, BlasOperation.Transpose);
            
            Assert.Equal(gemm, op);
        }
    }
}
