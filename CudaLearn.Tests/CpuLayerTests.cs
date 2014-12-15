using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaLearn.Tests
{
    public class CpuLayerTests
    {
        protected CpuLayerTests()
        {
            Context.Instance.Mode = ExecutionModeType.Cpu;
        }
    }
}
