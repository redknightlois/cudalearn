using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class BlobTests
    {
        [Fact]
        public void Initialization ()
        {
            var blob = new Blob();
            Assert.Equal(0, blob.Num);
            Assert.Equal(0, blob.Count);
            Assert.Equal(0, blob.Channels);
            Assert.Equal(0, blob.Height);
            Assert.Equal(0, blob.Width);

            Assert.Throws<InvalidOperationException>(() => blob.Data);
            Assert.Throws<InvalidOperationException>(() => blob.Diff);

            var preshapedBlob = new Blob(2, 3, 4, 5);
            Assert.Equal(2, preshapedBlob.Num);
            Assert.Equal(3, preshapedBlob.Channels);
            Assert.Equal(4, preshapedBlob.Height);
            Assert.Equal(5, preshapedBlob.Width);
            Assert.Equal(120, preshapedBlob.Count);

            Assert.NotNull(preshapedBlob.Data);
            Assert.NotNull(preshapedBlob.Diff);
            Assert.Equal(preshapedBlob.Count, preshapedBlob.Data.Count);
            Assert.Equal(preshapedBlob.Count, preshapedBlob.Diff.Count);
        }

        [Fact]
        public void Reshape()
        {
            var blob = new Blob();
            blob.Reshape(2, 3, 4, 5);
            Assert.Equal(2, blob.Num);
            Assert.Equal(3, blob.Channels);
            Assert.Equal(4, blob.Height);
            Assert.Equal(5, blob.Width);
            Assert.Equal(120, blob.Count);

            Assert.NotNull(blob.Data);
            Assert.NotNull(blob.Diff);
            Assert.Equal(blob.Count, blob.Data.Count);
            Assert.Equal(blob.Count, blob.Diff.Count);
        }

        [Fact]
        public void ReshapeAs()
        {
            var blob = new Blob();
            var preshaped = new Blob(2, 3, 4, 5);

            blob.ReshapeAs(preshaped);
            Assert.Equal(2, blob.Num);
            Assert.Equal(3, blob.Channels);
            Assert.Equal(4, blob.Height);
            Assert.Equal(5, blob.Width);
            Assert.Equal(120, blob.Count);

            Assert.NotNull(blob.Data);
            Assert.NotNull(blob.Diff);
            Assert.Equal(blob.Count, blob.Data.Count);
            Assert.Equal(blob.Count, blob.Diff.Count);
        }
    }
}
