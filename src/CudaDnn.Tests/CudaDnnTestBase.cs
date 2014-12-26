using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaDnn.Tests
{
    public class CudaDnnTestBase
    {
        public CudaDnnTestBase()
        {
            Assert.True(Environment.Is64BitProcess, "Tests are being run as 32bits processes. CuDNN is not supported on 32bits. Change the setting in Test->Test Settings->Default Processor Architecture->x64." );
        }
    }
}
