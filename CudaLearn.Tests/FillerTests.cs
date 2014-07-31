using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{
    public class FillerTests
    {
        [Fact]
        public void ConstantFiller()
        {
            var blob = new Blob(2, 3, 4, 5);

            var config = new ConstantFillerConfiguration(10.0f);
            var filler = new ConstantFiller(config);
            filler.Fill(blob);

            int count = blob.Count;
            var data = blob.Data;
            for (int i = 0; i < count; i++)
                Assert.Equal(data[i], 10.0f);
        }

        public static IEnumerable<object[]> MinMaxParameters
        {
            get
            {
                return new[]
                {
                    new object[] { 0.0f, 1.0f },
                    new object[] { -1.0f, 1.0f },
                    new object[] { 1.0f, 10.0f },
                };
            }
        }

        [Theory, PropertyData("MinMaxParameters")]
        public void UniformFiller( float min, float max )
        {
            var blob = new Blob(2, 3, 4, 5);
            var config = new UniformFillerConfiguration(min, max);
            var filler = new UniformFiller(config);
            filler.Fill(blob);

            int count = blob.Count;
            var data = blob.Data;
            for (int i = 0; i < count; i++)
            {
                Assert.True(data[i] >= min);
                Assert.True(data[i] <= max);
            }                
        }

        [Fact]
        public void PositiveUnitballFiller()
        {
            var blob = new Blob(2, 3, 4, 5);
            var filler = new PositiveUnitballFiller();
            filler.Fill(blob);

            int num = blob.Num;
            int count = blob.Count;
            int dim = count / num;

            var data = blob.Data;
            for (int i = 0; i < count; i++)
            {
                Assert.True(data[i] >= 0.0f);
                Assert.True(data[i] <= 1.0f);
            } 

            for ( int i = 0; i < num; i++ )
            {
                float sum = 0;
                for (int j = 0; j < dim; j++)
                    sum += blob.DataAt(i * dim + j);

                Assert.True(sum >= 0.999f);
                Assert.True(sum <= 1.001f);
            }
        }


        public static IEnumerable<object[]> MeanStdParameters
        {
            get
            {
                return new[]
                {
                    new object[] { 10.0f, 0.1f },
                    new object[] { -1.0f, 1.0f },
                    new object[] { 0.0f, 1.0f },
                };
            }
        }

        [Theory, PropertyData("MeanStdParameters")]
        public void GaussianDenseFiller(float meanParam, float stdParam)
        {
            var blob = new Blob(2, 3, 4, 5);
            var config = new GaussianFillerConfiguration(meanParam, stdParam);
            var filler = new GaussianFiller(config);
            filler.Fill(blob);

            float mean = 0;
            float var = 0;

            int count = blob.Count;

            for (int i = 0; i < count; i++)
            {
                mean += blob.DataAt(i);
                var += (blob.DataAt(i) - config.Mean) * (blob.DataAt(i) - config.Mean);
            }

            mean /= count;
            var /= count;

            Assert.True(mean >= config.Mean - config.Std * 5);
            Assert.True(mean <= config.Mean + config.Std * 5);

            float targetVar = config.Std * config.Std;
            Assert.True(var >= (targetVar / 5.0f));
            Assert.True(var <= (targetVar * 5.0f));
        }

        [Theory, PropertyData("MeanStdParameters")]
        public void GaussianSparseFiller(float meanParam, float stdParam)
        {
            var blob = new Blob(2, 3, 4, 5);
            var config = new GaussianFillerConfiguration(meanParam, stdParam) { IsSparse = true };

            var filler = new GaussianFiller(config);
            filler.Fill(blob);

            float mean = 0;
            float var = 0;

            int count = blob.Count;
            int zeroes = 0;

            for (int i = 0; i < count; i++)
            {
                if (blob.DataAt(i) == 0.0f)
                {
                    zeroes++;
                }
                else
                {
                    mean += blob.DataAt(i);
                    var += (blob.DataAt(i) - config.Mean) * (blob.DataAt(i) - config.Mean);
                }                    
            }

            mean /= (count - zeroes);
            var /= (count - zeroes);

            Assert.True(mean >= config.Mean - config.Std * 5);
            Assert.True(mean <= config.Mean + config.Std * 5);

            float targetVar = config.Std * config.Std;
            Assert.True(var >= (targetVar / 5.0f));
            Assert.True(var <= (targetVar * 5.0f));
        }

        [Fact]
        public void XavierFiller()
        {
            var blob = new Blob(2, 3, 4, 5);
            var filler = new XavierFiller();
            filler.Fill(blob);

            int fanIn = blob.Count / blob.Num;
            float scale = (float)Math.Sqrt(3 / fanIn);

            int count = blob.Count;
            var data = blob.Data;
            for (int i = 0; i < count; i++)
            {
                Assert.True(data[i] >= -scale);
                Assert.True(data[i] <= scale);
            }
        }
    }
}
