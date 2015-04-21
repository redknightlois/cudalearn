using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{

    public class InnerProductLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(2, 3, 4, 5);
        private readonly Tensor top = new Tensor();        

        public InnerProductLayerTests()
        {
            var filler = new UniformFiller();
            filler.Fill(bottom);
        }

        [Fact]
        public void InnerProductLayer_Setup()
        {
            var config = new InnerProductLayerConfiguration(10);

            var layer = new InnerProductLayer(config);
            layer.Setup(bottom, top);

            Assert.Equal(2, top.Num);
            Assert.Equal(10, top.Channels);
            Assert.Equal(1, top.Height);
            Assert.Equal(1, top.Width);
        }

        [Fact]
        public void InnerProductLayer_Forward()
        {
            var weightsFiller = new UniformFillerConfiguration(0, 1);
            var biasFiller = new UniformFillerConfiguration(1, 2);

            var config = new InnerProductLayerConfiguration(10, true, weightsFiller, biasFiller);

            var layer = new InnerProductLayer(config);
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            using (var topCpu = top.OnCpu())
            {
                int count = top.Count;

                for (int i = 0; i < count; i++)
                    Assert.True(topCpu.DataAt(i) >= 1f);
            }
        }

        public static IEnumerable<object[]> Configurations
        {
            get
            {
                return new[]
                {                 
                    new object[] { 30, null, null },
                    new object[] { 10, new XavierFillerConfiguration(), null },
                    new object[] { 5, new ConstantFillerConfiguration(), null },
                    new object[] { 100, new XavierFillerConfiguration(), new GaussianFillerConfiguration() },
                    new object[] { 100, new XavierFillerConfiguration(), new XavierFillerConfiguration() },
                };
            }
        }

        [Theory, MemberData("Configurations")]
        public void InnerProductLayer_BackwardGradient(int output, FillerConfiguration weightsFiller, FillerConfiguration biasFiller)
        {
            bool useBias = biasFiller != null;

            var config = new InnerProductLayerConfiguration(output, useBias, weightsFiller, biasFiller);
            var layer = new InnerProductLayer(config);

            var checker = new GradientChecker(1e-4f, 1e-3f);
            checker.CheckExhaustive(layer, bottom, top);
        }
    }
}
