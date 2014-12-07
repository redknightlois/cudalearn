using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{

    public class TanhLayerTests
    {
        private readonly Blob bottom = new Blob(2, 3, 6, 5);
        private readonly Blob top = new Blob();

        public TanhLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }

        [Fact]
        public void TanhLayer_Setup()
        {
            var layer = new TanhLayer();
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(bottom.Height, top.Height);
            Assert.Equal(bottom.Width, top.Width);
        }

        [Fact]
        public void TanhLayer_Forward()
        {
            var layer = new TanhLayer();
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);
            int count = bottom.Count;

            for (int i = 0; i < bottom.Num; i++)
            {
                for (int j = 0; j < bottom.Channels; j++)
                {
                    for ( int k = 0; k <bottom.Height; k++)
                    {
                        for ( int l = 0; l < bottom.Width; l++)
                        {
                            var v = (Math.Exp(2 * bottom.DataAt(i, j, k, l)) - 1) / (Math.Exp(2 * bottom.DataAt(i, j, k, l)) + 1);
                            Assert.True(MathHelpers.Equality(top.DataAt(i, j, k, l), v, 1e-4f));
                        }
                    }
                }
            }
        }

        [Fact]
        public void TanhLayer_BackwardGradient()
        {
            var layer = new TanhLayer();

            var checker = new GradientChecker(1e-2f, 1e-3f, 1701, 0.0d, 0.01f);
            checker.CheckEltwise(layer, bottom, top);
        }
    }

}
