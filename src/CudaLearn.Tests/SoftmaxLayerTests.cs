using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{

    public class SoftmaxLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(2, 10, 1, 1);
        private readonly Tensor top = new Tensor();

        public SoftmaxLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }

        [Fact]
        public void SoftmaxLayer_Setup()
        {
            var layer = new SoftmaxLayer();
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(bottom.Height, top.Height);
            Assert.Equal(bottom.Width, top.Width);
        }

        [Fact]
        public void SoftmaxLayer_Forward()
        {
            var layer = new SoftmaxLayer();
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);

            using (var topCpu = top.OnCpu())
            using (var bottomCpu = bottom.OnCpu())
            {
                int count = bottom.Count;

                int num = bottom.Num;
                int channels = bottom.Channels;
                for (int i = 0; i < num; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < channels; j++)
                        sum += topCpu.DataAt(i, j, 0, 0);

                    Assert.True(sum >= 0.999);
                    Assert.True(sum <= 1.001);
                }

                for (int i = 0; i < num; i++)
                {
                    double scale = 0;
                    for (int j = 0; j < channels; j++)
                        scale += Math.Exp(bottomCpu.DataAt(i, j, 0, 0));

                    for (int j = 0; j < channels; j++)
                    {
                        Assert.True(topCpu.DataAt(i, j, 0, 0) + 1e-4f >= Math.Exp(bottomCpu.DataAt(i, j, 0, 0)) / scale);
                        Assert.True(topCpu.DataAt(i, j, 0, 0) - 1e-4f <= Math.Exp(bottomCpu.DataAt(i, j, 0, 0)) / scale);
                    }
                }
            }
        }

        [Fact]
        public void SoftmaxLayer_BackwardGradient()
        {
            var layer = new SoftmaxLayer();

            var checker = new GradientChecker(1e-2f, 1e-3f);
            checker.CheckExhaustive(layer, bottom, top);
        }
    }
}
