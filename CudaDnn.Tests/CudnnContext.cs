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
            using (var context = CudnnContext.Create())
            using (var tensor = CudnnContext.CreateTensor())
            using (var convolution = CudnnContext.CreateConvolution())
            using (var pooling = CudnnContext.CreatePooling())
            using (var filter = CudnnContext.CreateFilter())
            {
                Assert.True(context.IsInitialized);
                Assert.NotNull(tensor);
                Assert.NotNull(convolution);
                Assert.NotNull(pooling);
                Assert.NotNull(filter);
            }
        }

    }
}
