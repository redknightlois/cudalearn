using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{
    public class PowerLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(2, 3, 6, 5);
        private readonly Tensor top = new Tensor();

        public PowerLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }


        [Fact]
        public void PowerLayer_Setup()
        {
            var layer = new ReluLayer();
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(bottom.Height, top.Height);
            Assert.Equal(bottom.Width, top.Width);
        }

        public static IEnumerable<object[]> Samples
        {
            get
            {
                return new[]
                {    
                    //new object[] { 0.37f, 0.83f, 0f }, // This case is numerically unstable with doubles.
                    //new object[] { 0.95f, 0.83f, 0f }, // This case is numerically unstable with doubles.
                    new object[] { 1f, 0.83f, 0f },
                    new object[] { 2f, 0.83f, 0f },                         
                    new object[] { 2f, 0.34f, -2.4f },
                    new object[] { 0.37f, 0.83f, -2.4f },                                                       
                    new object[] { 1f, 0.83f, -2.4f },      
                    new object[] { 2f, 0.83f, -2.4f },
                    new object[] { 2f, 0.5f, -2.4f },
                    new object[] { 8f, 0.5f, -2.4f },
                    new object[] { 0f, 0.83f, -2.4f },
                };
            }
        }

        [Theory, MemberData("Samples")]
        public void PowerLayer_Forward(double power, double scale, double shift)
        {
            var config = new PowerLayerConfiguration(power, scale, shift);
            var layer = new PowerLayer(config);
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            using (var topCpu = top.OnCpu())
            using (var bottomCpu = bottom.OnCpu())
            {
                double minPrecision = 1e-5f;

                // Now, check values
                int count = bottomCpu.Data.Count;
                for (int i = 0; i < count; i++)
                {
                    var expectedValue = Math.Pow(shift + scale * bottomCpu.DataAt(i), power);
                    if (power == 0 || power == 1 || power == 2)
                        Assert.False(double.IsNaN(topCpu.DataAt(i)));

                    if (double.IsNaN(expectedValue))
                        Assert.True(double.IsNaN(topCpu.DataAt(i)));
                    else
                    {
                        double precision = Math.Max(Math.Abs(expectedValue * 1e-4f), minPrecision);
                        Assert.True(MathHelpers.Equality(expectedValue, topCpu.DataAt(i), precision));
                    }
                }
            }
        }

        [Theory, MemberData("Samples")]
        public void PowerLayer_Backward(double power, double scale, double shift)
        {
            var config = new PowerLayerConfiguration(power, scale, shift);
            var layer = new PowerLayer(config);
            
            if ( power != 0 && power != 1 && power != 2 )
            {
                var minValue = -shift / scale;

                using (var bottomCpu = bottom.OnCpu())
                {
                    var bottomData = bottomCpu.Data;
                    for (int i = 0; i < bottom.Count; i++)
                    {
                        if (bottomData[i] < minValue)
                            bottomData[i] = minValue + (minValue - bottomData[i]);
                    }
                }
            }

            var checker = new GradientChecker(1e-2f, 1e-2f, 1701, 0.0d, 0.01f);
            checker.CheckEltwise(layer, bottom, top);
        }
    }
}
