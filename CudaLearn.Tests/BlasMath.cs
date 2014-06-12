using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class BlasMathTests
    {
        [Fact]
        public void AbsoluteSumSuccessful()
        {
            CudaLearnModule.Initialize();
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

            double gpuSum = BlasMath.AbsoluteSum(m);

            double cpuSum = 0;
            for (int j = 0; j < columns; j++)
            {
                double aux = mc[0, j];
                cpuSum += aux >= 0 ? aux : -aux;
            }

            Assert.InRange(gpuSum - cpuSum, -Matrix<double>.DefaultEpsilon, Matrix<double>.DefaultEpsilon);
        }
    }
}
