using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaDnn.Tests
{
    public class CudnnContextTests : CudaDnnTestBase
    {

        [Fact]
        public void Lifecycle ()
        {
            using ( var context = CudnnContext.Create() )
            {                
            }
        }

    }
}
