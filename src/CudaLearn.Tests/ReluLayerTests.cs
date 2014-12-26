using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class ReluLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(2, 3, 6, 5);
        private readonly Tensor top = new Tensor();

        public ReluLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }


        [Fact]
        public void ReluLayer_Setup()
        {
            var layer = new ReluLayer();
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(bottom.Height, top.Height);
            Assert.Equal(bottom.Width, top.Width);
        }

        [Fact]
        public void ReluLayer_Forward()
        {
            var layer = new ReluLayer();
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
                    Assert.True(topCpu.DataAt(i) == 0.0d || topCpu.DataAt(i) == bottomCpu.DataAt(i));
                };
            }
        }

        [Fact]
        public void ReluLayer_BackwardGradient()
        {
            var layer = new ReluLayer();

            var checker = new GradientChecker(1e-2f, 1e-3f, 1701, 0.0d, 0.01f);
            checker.CheckEltwise(layer, bottom, top);
        }

        [Fact]
        public void ReluLayer_ForwardWithLeakyUnits()
        {
            // http://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs

            var config = new ReluLayerConfiguration(0.01f);
            var layer = new ReluLayer(config);
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            double slope = layer.Parameters.NegativeSlope;

            Assert.Equal(bottom.Count, top.Count);

            using (var topCpu = top.OnCpu())
            using (var bottomCpu = bottom.OnCpu())
            {
                int count = bottom.Count;
                for (int i = 0; i < count; i++)
                {
                    if (bottomCpu.DataAt(i) <= 0)
                    {
                        Assert.True(topCpu.DataAt(i) >= bottomCpu.DataAt(i) * slope - 0.000001);
                    }
                    else
                    {
                        Assert.True(topCpu.DataAt(i) >= 0.0d);
                        Assert.True(topCpu.DataAt(i) == 0.0d || topCpu.DataAt(i) == bottomCpu.DataAt(i));
                    }
                };
            }
        }

        [Fact]
        public void ReluLayer_ForwardGradientWithLeakyUnits()
        {
            // http://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs

            var config = new ReluLayerConfiguration(0.01f);
            var layer = new ReluLayer(config);
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            double slope = layer.Parameters.NegativeSlope;

            Assert.Equal(bottom.Count, top.Count);

            using (var topCpu = top.OnCpu())
            using (var bottomCpu = bottom.OnCpu())
            {
                int count = bottom.Count;
                for (int i = 0; i < count; i++)
                {
                    if (bottomCpu.DataAt(i) <= 0)
                    {
                        Assert.True(topCpu.DataAt(i) >= bottomCpu.DataAt(i) * slope - 0.000001);
                    }
                    else
                    {
                        Assert.True(topCpu.DataAt(i) >= 0.0d);
                        Assert.True(topCpu.DataAt(i) == 0.0d || topCpu.DataAt(i) == bottomCpu.DataAt(i));
                    }
                };
            }
        }

        [Fact]
        public void ReluLayer_BackwardGradientWithLeakyUnits()
        {
            var config = new ReluLayerConfiguration(0.01f);
            var layer = new ReluLayer(config);

            var checker = new GradientChecker(1e-2f, 1e-2f, 1701, 0.0d, 0.01f);
            checker.CheckEltwise(layer, bottom, top);
        }
    }
}
