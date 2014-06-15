using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn.Tests
{
    public class MathTestsBase : IDisposable
    {
        public MathTestsBase()
        {
            CudaLearnModule.Initialize();
        }

        protected bool EqualsWithEpsilon(float a, float b)
        {
            return Math.Abs(a - b) < Matrix<float>.Epsilon;
        }

        public void Dispose()
        {
            CudaLearnModule.Release();
        }
    }
}
