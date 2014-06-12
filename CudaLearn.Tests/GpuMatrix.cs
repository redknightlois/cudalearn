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
            CudaLearnModule.AllowHandyForDebugButVerySlowGpuMemoryAccess = true;
        }

        [Fact]
        public void CreateGpuMatrixOfTWithNumericTypes()
        {
            CudaLearnModule.Initialize();

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
            CudaLearnModule.Initialize();

            Assert.Throws<NotSupportedException>(() => new GpuMatrix<byte>(2, 2));
            Assert.Throws<NotSupportedException>(() => new GpuMatrix<SampleStruct>(2, 2));
            Assert.Throws<NotSupportedException>(() => new GpuMatrix<sbyte>(2, 2));
        }
    }
}
