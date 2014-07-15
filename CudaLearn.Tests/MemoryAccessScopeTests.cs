using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class MemoryAccessScopeTests : MathTestsBase
    {

        [Fact]
        public void EnsureMatrixIsLocked()
        {
            var m = new GpuMatrix<float>(2, 2);

            using (var s = new MemoryAccessScope<float>(m))
            {
                Assert.Throws<InvalidOperationException>(() => m.Clone());
                Assert.Throws<InvalidOperationException>(() => m + m);
                Assert.Throws<InvalidOperationException>(() => m * m);
                Assert.Throws<InvalidOperationException>(() => m - m);
                Assert.Throws<InvalidOperationException>(() => m + 5);
                Assert.Throws<InvalidOperationException>(() => m - 5);
                Assert.Throws<InvalidOperationException>(() => -m);
                Assert.Throws<InvalidOperationException>(() => m.Dispose());
                Assert.Throws<InvalidOperationException>(() => m.ToString());
            }

            Assert.DoesNotThrow(() => m.Clone());            
            Assert.DoesNotThrow(() => m.ToString());
        }

        [Fact]
        public void BlindWriteSuccessful()
        {
            var m = new GpuMatrix<float>(2, 2);

            using (var s = new MemoryAccessScope<float>(m))
            {
                s[0, 0] = 1.0f;
                s[0, 1] = 1.0f;
                s[1, 0] = 1.0f;
                s[1, 1] = 1.0f;
            }

            Assert.Equal(new Matrix<float>(2, 2, 1.0f), m);
        }

        [Fact]
        public void ReadSuccessful()
        {
            var m = GpuMatrix<int>.Identity(2);

            using (var s = new MemoryAccessScope<int>(m))
            {
                Assert.Equal(1, s[0, 0]);
                Assert.Equal(1, s[1, 1]);
                Assert.Equal(0, s[1, 0]);
                Assert.Equal(0, s[0, 1]);
            }            
        }
    }
}
