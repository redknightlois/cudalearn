using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class MixedMatrixOperationsTests : MathTestsBase
    {

        [Fact]
        public void OperationsInDeviceWithCopyToHostCasting ()
        {
            int size = 100;

            var generator = new Random(123);

            var c = new Matrix<double>(size, size);
            for (int i = 0; i < c.Rows; i++)
                for (int j = 0; j < c.Columns; j++)
                    c[i, j] = generator.Next(150) / 150.0f;

            Matrix<double> c1;
            using ( var g = GpuMatrix<double>.Identity(size) )
            using ( var r = g * c * g )
            {
                c1 = (Matrix<double>)r;
            }

            // We perform all tests in the Host (CPU). 
            Assert.Equal<Matrix<double>>(c1, c);
        }
    }
}
