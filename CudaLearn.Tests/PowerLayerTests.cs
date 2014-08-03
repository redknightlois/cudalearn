using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{
    public class PowerLayerTests
    {
        private readonly Blob bottom = new Blob(2, 3, 6, 5);
        private readonly Blob top = new Blob();

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
                    //new object[] { 0.37f, 0.83f, 0f },
                    new object[] { 0.90f, 0.83f, 0f },
                    //new object[] { 1f, 0.83f, 0f },
                    //new object[] { 2f, 0.83f, 0f },                         
                    //new object[] { 2f, 0.34f, -2.4f },
                    //new object[] { 0.37f, 0.83f, -2.4f },                                                       
                    //new object[] { 1f, 0.83f, -2.4f },      
                    //new object[] { 2f, 0.83f, -2.4f },
                    //new object[] { 2f, 0.5f, -2.4f },
                    //new object[] { 0f, 0.83f, -2.4f },
                };
            }
        }

        [Theory, PropertyData("Samples")]
        public void PowerLayer_Forward(float power, float scale, float shift)
        {
            var config = new PowerLayerConfiguration(power, scale, shift);
            var layer = new PowerLayer(config);
            layer.Setup(bottom, top);
            layer.Forward(bottom, top);

            float minPrecision = 1e-5f;

            // Now, check values
            int count = bottom.Data.Count;
            for (int i = 0; i < count; i++)
            {
                var expectedValue = (float)Math.Pow(shift + scale * bottom.DataAt(i), power);
                if (power == 0 || power == 1 || power == 2)
                    Assert.False(float.IsNaN(top.DataAt(i)));

                if (float.IsNaN(expectedValue))
                    Assert.True(float.IsNaN(top.DataAt(i)));
                else
                {
                    float precision = Math.Max(Math.Abs(expectedValue * 1e-4f), minPrecision);
                    Assert.True(MathHelpers.Equality(expectedValue, top.DataAt(i), precision));
                }
            }
        }

        [Theory, PropertyData("Samples")]
        public void PowerLayer_Backward(float power, float scale, float shift)
        {
            var config = new PowerLayerConfiguration(power, scale, shift);
            var layer = new PowerLayer(config);

            if ( power != 0 && power != 1 && power != 2 )
            {
                var minValue = -shift / scale;
                
                var bottomData = bottom.Data;
                for ( int i = 0; i < bottom.Count; i++ )
                {
                    if (bottomData[i] < minValue)
                        bottomData[i] = minValue + (minValue - bottomData[i]);
                }
            }

            var checker = new GradientChecker(1e-2f, 1e-2f, 1701, 0.0f, 0.01f);
            checker.CheckEltwise(layer, bottom, top);
        }
    }
}
