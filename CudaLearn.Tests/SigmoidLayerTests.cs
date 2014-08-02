using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{

    public class SigmoidLayerTests
    {
        private readonly Blob bottom = new Blob(2, 3, 6, 5);
        private readonly Blob top = new Blob();

        public SigmoidLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }

        [Fact]
        public void SigmoidLayer_Setup()
        {
            var layer = new SigmoidLayer();
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(bottom.Height, top.Height);
            Assert.Equal(bottom.Width, top.Width);
        }

        [Fact]
        public void SigmoidLayer_Forward()
        {
            var layer = new SigmoidLayer();
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);
            int count = bottom.Count;
            for (int i = 0; i < count; i++)
            {
                Assert.True(MathHelpers.Equality(top.DataAt(i), 1.0f / (1.0f + (float)Math.Exp(-bottom.DataAt(i)))));

                // check that we squashed the value between 0 and 1
                Assert.True(top.DataAt(i) >= 0.0f);
                Assert.True(top.DataAt(i) <= 1.0f);
            };
        }

        [Fact]
        public void SigmoidLayer_BackwardGradient()
        {
            var layer = new SigmoidLayer();

            var checker = new GradientChecker(1e-2f, 1e-3f, 1701, 0.0f, 0.01f);
            checker.CheckEltwise(layer, bottom, top);
        }
    }
}
