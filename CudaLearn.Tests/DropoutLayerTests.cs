using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class DropoutLayerTests
    {
        private readonly Blob bottom = new Blob(2, 3, 6, 5);
        private readonly Blob top = new Blob();

        public DropoutLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);                        
        }


        [Fact]
        public void DropoutLayer_Setup()
        {
            var layer = new DropoutLayer();
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(bottom.Height, top.Height);
            Assert.Equal(bottom.Width, top.Width);
        }

        [Fact]
        public void DropoutLayer_ForwardTrainPhase()
        {
            Context.Instance.Phase = PhaseType.Train;

            var layer = new DropoutLayer();
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);

            float scale = 1f / (1f - layer.Parameters.Ratio);

            int count = bottom.Count;
            for (int i = 0; i < count; i++)
            {
                if (!MathHelpers.Equality(top.DataAt(i), 0))
                    Assert.True(MathHelpers.Equality(top.DataAt(i), bottom.DataAt(i) * scale));
            };
        }

        [Fact]
        public void DropoutLayer_ForwardTestPhase()
        {
            Context.Instance.Phase = PhaseType.Test;

            var layer = new DropoutLayer();
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);

            int count = bottom.Count;
            for (int i = 0; i < count; i++)
            {
                if (!MathHelpers.Equality(top.DataAt(i), 0))
                    Assert.True(MathHelpers.Equality(top.DataAt(i), bottom.DataAt(i)));
            };
        }

        [Fact]
        public void DropoutLayer_BackwardGradientTrainPhase()
        {
            Context.Instance.Phase = PhaseType.Train;

            var layer = new DropoutLayer();

            var checker = new GradientChecker(1e-2f, 1e-3f);
            checker.CheckEltwise(layer, bottom, top);
        }

        [Fact]
        public void DropoutLayer_BackwardGradientTestPhase()
        {
            Context.Instance.Phase = PhaseType.Test;

            var layer = new DropoutLayer();

            var checker = new GradientChecker(1e-2f, 1e-3f);
            checker.CheckEltwise(layer, bottom, top);
        }


    }
}
