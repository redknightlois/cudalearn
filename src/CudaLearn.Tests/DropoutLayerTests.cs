﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{
    public class DropoutLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(2, 3, 6, 5);
        private readonly Tensor top = new Tensor();

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

        public static IEnumerable<object[]> DropoutParameter
        {
            get
            {
                return new[]
                {    
                    new object[] { 0.10f},
                    new object[] { 0.25f },
                    new object[] { 0.5f },                         
                    new object[] { 0.75f },
                    new object[] { 0.9f },                                                       
                };
            }
        }

        [Theory, MemberData("DropoutParameter")]
        public void DropoutLayer_ForwardTrainPhase(double ratio)
        {
            Context.Instance.Phase = PhaseType.Train;

            var config = new DropoutLayerConfiguration(ratio);
            var layer = new DropoutLayer(config);
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);

            using (var topCpu = top.OnCpu())
            using (var bottomCpu = bottom.OnCpu())
            {
                double scale = 1f / (1f - layer.Parameters.Ratio);

                int count = bottom.Count;
                int kept = 0;
                for (int i = 0; i < count; i++)
                {
                    if (!MathHelpers.Equality(topCpu.DataAt(i), 0))
                    {
                        kept++;
                        Assert.True(MathHelpers.Equality(topCpu.DataAt(i), bottomCpu.DataAt(i) * scale));
                    }
                };

                double stdError = Math.Sqrt(ratio * (1 - ratio) / count);
                double empiricalDropoutRatio = 1.0d - ((double)kept / count);

                Assert.True(MathHelpers.Equality(ratio, empiricalDropoutRatio, 1.96 * stdError));
            } 
        }

        [Fact]
        public void DropoutLayer_ForwardTestPhase()
        {
            Context.Instance.Phase = PhaseType.Test;

            var layer = new DropoutLayer();
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            Assert.Equal(bottom.Count, top.Count);

            using (var topCpu = top.OnCpu())
            using (var bottomCpu = bottom.OnCpu())
            {
                int count = bottom.Count;
                for (int i = 0; i < count; i++)
                {
                    if (!MathHelpers.Equality(topCpu.DataAt(i), 0))
                        Assert.True(MathHelpers.Equality(topCpu.DataAt(i), bottomCpu.DataAt(i)));
                };
            }
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
