using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class MatrixTest 
    {
        [Fact]
        public void SumVectorOverAxis()
        {
            var m1 = Matrix<float>.Build.Dense(2, 3);

            // 1 0 2
            // 0 3 0            
            m1[0, 0] = 1;
            m1[0, 1] = 0;
            m1[0, 2] = 2;
            m1[1, 0] = 0;
            m1[1, 1] = 3;
            m1[1, 2] = 0;

            // 1 0 2             2 2 5
            // 0 3 0  + 1 2 3 =  1 5 3

            var result = Matrix<float>.Build.Dense(2, 3);
            result[0, 0] = 2;
            result[0, 1] = 2;
            result[0, 2] = 5;
            result[1, 0] = 1;
            result[1, 1] = 5;
            result[1, 2] = 3;

            var columnVector = Vector<float>.Build.Dense(3, 1);
            columnVector[1] = 2;
            columnVector[2] = 3;

            var sumCols = Functions.AddVectorOnEachColumn( m1, columnVector );

            Assert.Equal(result, sumCols);

            // 1 0 2    1    2 1 3
            // 0 3 0  + 1 =  1 4 1

            result[0, 0] = 2;
            result[0, 1] = 1;
            result[0, 2] = 3;
            result[1, 0] = 1;
            result[1, 1] = 4;
            result[1, 2] = 1;

            var rowsVector = Vector<float>.Build.Dense(2, 1);
            var sumRows = Functions.AddVectorOnEachRow(m1, rowsVector);

            Assert.Equal(result, sumRows);
        }
    }
}
