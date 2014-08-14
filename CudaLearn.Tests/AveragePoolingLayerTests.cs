using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{
    public class AveragePoolingLayerTests
    {
        private readonly Blob bottom = new Blob(2, 3, 6, 5);
        private readonly Blob top = new Blob();

        public AveragePoolingLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }

        [Fact]
        public void AveragePoolingLayer_Setup()
        {            
            var layer = new AveragePoolingLayer(3, 2);
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(3, top.Height);
            Assert.Equal(2, top.Width);
        }

        [Fact]
        public void AveragePoolingLayer_SetupPadded()
        {
            var layer = new AveragePoolingLayer(3, 2, 1);
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(4, top.Height);
            Assert.Equal(3, top.Width);
        }

        [Fact]
        public void AveragePoolingLayer_Forward()
        {
            var bottom = new Blob(1, 1, 3, 3);

            var filler = new ConstantFiller(2.0f);
            filler.Fill(bottom);

            var layer = new AveragePoolingLayer(3, 1, 1);
            layer.Setup(bottom, top);

            Assert.Equal(1, top.Num);
            Assert.Equal(1, top.Channels);
            Assert.Equal(3, top.Height);
            Assert.Equal(3, top.Width);

            layer.Forward(bottom, top);

            var topData = top.Data;
            AssertInRange(8.0f / 9, topData[0]);
            AssertInRange(4.0f / 3, topData[1]);
            AssertInRange(8.0f / 9, topData[2]);
            AssertInRange(4.0f / 3, topData[3]);
            AssertInRange(2.0f, topData[4]);
            AssertInRange(4.0f / 3, topData[5]);
            AssertInRange(8.0f / 9, topData[6]);
            AssertInRange(4.0f / 3, topData[7]);
            AssertInRange(8.0f / 9, topData[8]);
        }

        public static IEnumerable<object[]> Configurations
        {
            get
            {
                return new[]
                {                 
                    new object[] { new Size(1,1), new Size(1,1), new Size() },
                    new object[] { new Size(2,2), new Size(2,2), new Size() },
                    new object[] { new Size(1,2), new Size(1,1), new Size() },
                    new object[] { new Size(3,3), new Size(2,2), new Size() },
                    new object[] { new Size(3,4), new Size(2,2), new Size() },
                    new object[] { new Size(4,4), new Size(2,2), new Size() },
                    new object[] { new Size(4,3), new Size(2,2), new Size() },
                    new object[] { new Size(4,4), new Size(2,2), new Size(1,1) },
                };
            }
        }

        [Theory, PropertyData("Configurations")]
        public void AveragePoolingLayer_BackwardGradient( Size kernel, Size stride, Size padding )
        {
            var filler = new ConstantFiller(2.0f);
            filler.Fill(bottom);

            var checker = new GradientChecker(1e-2f, 1e-2f);

            var layer = new AveragePoolingLayer(kernel, stride, padding);
            checker.CheckExhaustive(layer, bottom, top);
        }
        

        protected void AssertInRange(float expected, float value, float epsilon = 1e-5f)
        {
            float v = expected - value;
            Assert.InRange(v, -epsilon, +epsilon);
        }
    }
}
