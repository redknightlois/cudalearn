using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using CudaDnn.Impl;

namespace CudaDnn.Tests
{
    public class CudnnContextTests : CudaDnnTestBase
    {

        [Fact]
        public void Lifecycle()
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

        [Fact]
        public void ContextWithStream()
        {
            using ( var cuda = new CudaContext() )
            using ( var stream = new CudaStream(CUStreamFlags.Default) )
            {
                using (var context = CudnnContext.Create(stream))
                {
                    Assert.True(context.IsInitialized);

                    var streamId = default (CUstream);
                    CudnnContext.Invoke(() => CudnnNativeMethods.cudnnGetStream(context.Handle, out streamId));

                    Assert.Equal(stream.Stream, streamId);
                }
            }            
        }

    }
}
