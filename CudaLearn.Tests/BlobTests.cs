using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class BlobTests : CpuLayerTests
    {
        [Fact]
        public void Blob_Initialization()
        {
            var blob = new Tensor();
            Assert.Equal(0, blob.Num);
            Assert.Equal(0, blob.Count);
            Assert.Equal(0, blob.Channels);
            Assert.Equal(0, blob.Height);
            Assert.Equal(0, blob.Width);

            Assert.Throws<InvalidOperationException>(() => blob.OnCpu());
            // Assert.Throws<InvalidOperationException>(() => blob.Data);
            // Assert.Throws<InvalidOperationException>(() => blob.Diff);

            var preshapedBlob = new Tensor(2, 3, 4, 5);
            Assert.Equal(2, preshapedBlob.Num);
            Assert.Equal(3, preshapedBlob.Channels);
            Assert.Equal(4, preshapedBlob.Height);
            Assert.Equal(5, preshapedBlob.Width);
            Assert.Equal(120, preshapedBlob.Count);

            using (var preshapedBlobCpu = preshapedBlob.OnCpu())
            {
                Assert.NotNull(preshapedBlobCpu.Data);
                Assert.NotNull(preshapedBlobCpu.Diff);
                Assert.Equal(preshapedBlob.Count, preshapedBlobCpu.Data.Count);
                Assert.Equal(preshapedBlob.Count, preshapedBlobCpu.Diff.Count);
            }
        }

        [Fact]
        public void Blob_Reshape()
        {
            var blob = new Tensor();
            blob.Reshape(2, 3, 4, 5);
            Assert.Equal(2, blob.Num);
            Assert.Equal(3, blob.Channels);
            Assert.Equal(4, blob.Height);
            Assert.Equal(5, blob.Width);
            Assert.Equal(120, blob.Count);

            using (var blobCpu = blob.OnCpu())
            {
                Assert.NotNull(blobCpu.Data);
                Assert.NotNull(blobCpu.Diff);
                Assert.Equal(blobCpu.Count, blobCpu.Data.Count);
                Assert.Equal(blobCpu.Count, blobCpu.Diff.Count);
            }
        }

        [Fact]
        public void Blob_ReshapeAs()
        {
            var blob = new Tensor();
            var preshaped = new Tensor(2, 3, 4, 5);

            blob.ReshapeAs(preshaped);
            Assert.Equal(2, blob.Num);
            Assert.Equal(3, blob.Channels);
            Assert.Equal(4, blob.Height);
            Assert.Equal(5, blob.Width);
            Assert.Equal(120, blob.Count);

            using (var blobCpu = blob.OnCpu())
            {
                Assert.NotNull(blobCpu.Data);
                Assert.NotNull(blobCpu.Diff);
                Assert.Equal(blobCpu.Count, blobCpu.Data.Count);
                Assert.Equal(blobCpu.Count, blobCpu.Diff.Count);
            }
        }
    }
}
