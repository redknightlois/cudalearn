using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class ReluLayerTests
    {
        private readonly Blob bottom = new Blob(2, 3, 6, 5);
        private readonly Blob top = new Blob();

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
            int count = bottom.Count;
            for (int i = 0; i < count; i++)
            {
                Assert.True(top.DataAt(i) >= 0.0f);
                Assert.True(top.DataAt(i) == 0.0f || top.DataAt(i) == bottom.DataAt(i));
            };
        }

        [Fact]
        public void ReluLayer_Gradient()
        {
            var layer = new ReluLayer();

            var checker = new GradientChecker(1e-2f, 1e-3f, 1701, 0.0f, 0.01f);
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

            float slope = layer.Parameters.NegativeSlope;

            Assert.Equal(bottom.Count, top.Count);
            int count = bottom.Count;
            for (int i = 0; i < count; i++)
            {
                if (bottom.DataAt(i) <= 0)
                {
                    Assert.True(top.DataAt(i) >= bottom.DataAt(i) * slope - 0.000001);
                }
                else
                {
                    Assert.True(top.DataAt(i) >= 0.0f);
                    Assert.True(top.DataAt(i) == 0.0f || top.DataAt(i) == bottom.DataAt(i));
                }
            };
        }

        [Fact]
        public void ReluLayer_GradientWithLeakyUnits()
        {
            var config = new ReluLayerConfiguration(0.01f);
            var layer = new ReluLayer(config);

            var checker = new GradientChecker(1e-2f, 1e-2f, 1701, 0.0f, 0.01f);
            checker.CheckEltwise(layer, bottom, top);
        }

        [Fact]
        public void ReluLayer_ForwardGradientWithLeakyUnits()
        {
            // http://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs

            var config = new ReluLayerConfiguration(0.01f);
            var layer = new ReluLayer(config);
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            float slope = layer.Parameters.NegativeSlope;

            Assert.Equal(bottom.Count, top.Count);
            int count = bottom.Count;
            for (int i = 0; i < count; i++)
            {
                if (bottom.DataAt(i) <= 0)
                {
                    Assert.True(top.DataAt(i) >= bottom.DataAt(i) * slope - 0.000001);
                }
                else
                {
                    Assert.True(top.DataAt(i) >= 0.0f);
                    Assert.True(top.DataAt(i) == 0.0f || top.DataAt(i) == bottom.DataAt(i));
                }
            };
        }
    }
}
