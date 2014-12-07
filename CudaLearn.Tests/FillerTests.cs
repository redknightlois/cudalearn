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
        public void Filler_Constant()
        {
            var blob = new Blob(2, 3, 4, 5);

            var config = new ConstantFillerConfiguration(10.0d);
            var filler = new ConstantFiller(config);
            filler.Fill(blob);

            int count = blob.Count;
            var data = blob.Data;
            for (int i = 0; i < count; i++)
                Assert.Equal(data[i], 10.0d);
        }

        public static IEnumerable<object[]> MinMaxParameters
        {
            get
            {
                return new[]
                {
                    new object[] { 0.0d, 1.0d },
                    new object[] { -1.0d, 1.0d },
                    new object[] { 1.0d, 10.0d },
                };
            }
        }

        [Theory, MemberData("MinMaxParameters")]
        public void Filler_Uniform(double min, double max)
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
        public void Filler_PositiveUnitball()
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
                Assert.True(data[i] >= 0.0d);
                Assert.True(data[i] <= 1.0d);
            } 

            for ( int i = 0; i < num; i++ )
            {
                double sum = 0;
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
                    new object[] { 10.0d, 0.1f },
                    new object[] { -1.0d, 1.0d },
                    new object[] { 0.0d, 1.0d },
                };
            }
        }

        [Theory, MemberData("MeanStdParameters")]
        public void Filler_GaussianDense(double meanParam, double stdParam)
        {
            var blob = new Blob(2, 3, 4, 5);
            var config = new GaussianFillerConfiguration(meanParam, stdParam);
            var filler = new GaussianFiller(config);
            filler.Fill(blob);

            double mean = 0;
            double var = 0;

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

            double targetVar = config.Std * config.Std;
            Assert.True(var >= (targetVar / 5.0d));
            Assert.True(var <= (targetVar * 5.0d));
        }

        [Theory, MemberData("MeanStdParameters")]
        public void Filler_GaussianSparse(double meanParam, double stdParam)
        {
            var blob = new Blob(2, 3, 4, 5);
            var config = new GaussianFillerConfiguration(meanParam, stdParam) { IsSparse = true };

            var filler = new GaussianFiller(config);
            filler.Fill(blob);

            double mean = 0;
            double var = 0;

            int count = blob.Count;
            int zeroes = 0;

            for (int i = 0; i < count; i++)
            {
                if (blob.DataAt(i) == 0.0d)
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

            double targetVar = config.Std * config.Std;
            Assert.True(var >= (targetVar / 5.0d));
            Assert.True(var <= (targetVar * 5.0d));
        }

        [Fact]
        public void Filler_Xavier()
        {
            var blob = new Blob(2, 3, 4, 5);
            var filler = new XavierFiller();
            filler.Fill(blob);

            int fanIn = blob.Count / blob.Num;
            double scale = Math.Sqrt(3 / fanIn);

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
