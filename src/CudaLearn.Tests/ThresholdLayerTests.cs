using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class ThresholdLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(2, 3, 6, 5);
        private readonly Tensor top = new Tensor();

        public ThresholdLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }

        [Fact]
        public void ThresholdLayer_Setup()
        {
            var layer = new ThresholdLayer();
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(bottom.Height, top.Height);
            Assert.Equal(bottom.Width, top.Width);
        }

        [Fact]
        public void ThresholdLayer_Forward()
        {
            var layer = new ThresholdLayer();
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);

            using (var topCpu = top.OnCpu())
            using (var bottomCpu = bottom.OnCpu())
            {
                int count = bottom.Count;
                for (int i = 0; i < count; i++)
                {
                    Assert.True(topCpu.DataAt(i) >= 0.0d);
                    Assert.True(topCpu.DataAt(i) <= 1.0d);

                    if (topCpu.DataAt(i) == 0.0d)
                    {
                        Assert.True(bottomCpu.DataAt(i) <= layer.Parameters.Threshold);
                    }
                    else if (topCpu.DataAt(i) == 1.0d)
                    {
                        Assert.True(bottomCpu.DataAt(i) > layer.Parameters.Threshold);
                    }
                    else Assert.True(false);
                };
            }
        }

        [Fact]
        public void ThresholdLayer_ForwardWithConfiguration()
        {
            var config = new ThresholdLayerConfiguration(0.5f);
            var layer = new ThresholdLayer(config);
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);

            using (var topCpu = top.OnCpu())
            using (var bottomCpu = bottom.OnCpu())
            {
                int count = bottom.Count;
                for (int i = 0; i < count; i++)
                {
                    Assert.True(topCpu.DataAt(i) >= 0.0d);
                    Assert.True(topCpu.DataAt(i) <= 1.0d);

                    if (topCpu.DataAt(i) == 0.0d)
                    {
                        Assert.True(bottomCpu.DataAt(i) <= layer.Parameters.Threshold);
                    }
                    else if (topCpu.DataAt(i) == 1.0d)
                    {
                        Assert.True(bottomCpu.DataAt(i) > layer.Parameters.Threshold);
                    }
                    else Assert.True(false);
                };
            }
        }

        [Fact]
        public void ThresholdLayer_Backward()
        {
            var layer = new ThresholdLayer();
            layer.Setup(bottom, top);
            Assert.Throws<NotSupportedException>(() => layer.Backward(top, new List<bool> { false }, bottom));
        }
    }
}
