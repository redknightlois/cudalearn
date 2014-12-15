using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class BnllLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(2, 3, 6, 5);
        private readonly Tensor top = new Tensor();

        public BnllLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }

        [Fact]
        public void BnllLayer_Setup()
        {
            var layer = new BnllLayer();
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(bottom.Height, top.Height);
            Assert.Equal(bottom.Width, top.Width);
        }

        [Fact]
        public void BnllLayer_Forward()
        {
            var layer = new BnllLayer();
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
                    Assert.True(topCpu.DataAt(i) >= bottomCpu.DataAt(i));
                };
            }
        }

        [Fact]
        public void BnllLayer_BackwardGradient()
        {
            var layer = new BnllLayer();

            var checker = new GradientChecker(1e-2f, 1e-3f);
            checker.CheckEltwise(layer, bottom, top);
        }
    }
}
