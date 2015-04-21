using CudaLearn.Providers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests.Providers
{
    public class VclHostTensorMathTests
    {
        [Fact]
        public void ViennaHostProvider_GetVersion()
        {
            var provider = new VclHostTensorMath();
            Assert.Equal(162, provider.GetVersion());
        }

        [Fact]
        public void ViennaHostProvider_Set()
        {
            var provider = new VclHostTensorMath();

            var y = new ArraySlice<double>(10);
            provider.Set(10.0d, y);

            var yInt = new ArraySlice<double>(10);
            provider.Set(10, yInt);

            Assert.All(y, yy => MathHelpers.Equality(yy, 10.0d));
            Assert.All(yInt, yyInt => MathHelpers.Equality(yyInt, 10.0d));
        }

        [Fact]
        public void ViennaHostProvider_Copy()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            provider.Set(10.0d, x);

            var y = new ArraySlice<double>(10);
            provider.Copy(x, y);

            Assert.All(y, yy => MathHelpers.Equality(yy, 10.0d));
        }

        [Fact]
        public void ViennaHostProvider_Add()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            provider.Set(10.0d, x);
            provider.Add(10.0d, x);

            Assert.All(x, xx => MathHelpers.Equality(xx, 20.0d));
        }

        [Fact]
        public void ViennaHostProvider_Axpby()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            provider.Set(10.0d, x);

            var y = new ArraySlice<double>(10);
            provider.Set(5.0d, y);

            provider.Axpby(1, x, 2, y);

            var z = new ArraySlice<double>(10);
            provider.Set(10.0d, z);

            provider.Axpy(2, x, z);

            Assert.All(y, yy => MathHelpers.Equality(yy, 20.0d));
            Assert.All(z, zz => MathHelpers.Equality(zz, 30.0d));
        }

        [Fact]
        public void ViennaHostProvider_Substract()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            provider.Set(10.0d, x);

            var y = new ArraySlice<double>(10);
            provider.Set(5.0d, y);

            provider.Substract(x, y);

            Assert.All(y, yy => MathHelpers.Equality(yy, 5.0d));
        }

        [Fact]
        public void ViennaHostProvider_Powx()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            var y = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = 2;
            }

            provider.Powx(x, y);
            provider.Powx(x, x);

            for (int i = 0; i < x.Length; i++)
            {
                Assert.True(MathHelpers.Equality(Math.Pow(i, i), x[i]));
                Assert.True(MathHelpers.Equality(Math.Pow(i, 2), y[i]));
            }
        }

        [Fact]
        public void ViennaHostProvider_Exp()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
                x[i] = i;

            var y = new ArraySlice<double>(10);
            provider.Exp(x, y);
            
            for (int i = 0; i < y.Length; i++)
                Assert.True(MathHelpers.Equality(Math.Exp(i), y[i]));
        }

        [Fact]
        public void ViennaHostProvider_Multiply()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            var y = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = 2;
            }

            provider.Multiply(x, y);

            for (int i = 0; i < x.Length; i++)
                Assert.True(MathHelpers.Equality(x[i] * 2, y[i]));
        }

        [Fact]
        public void ViennaHostProvider_Square()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            var y = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = 2;
            }

            provider.Square(x, y);

            for (int i = 0; i < x.Length; i++)
                Assert.True(MathHelpers.Equality(x[i] * x[i], y[i]));
        }

        [Fact]
        public void ViennaHostProvider_Divide()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            var y = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = 2;
            }

            provider.Divide(x, y);

            for (int i = 0; i < x.Length; i++)
                Assert.True(MathHelpers.Equality(x[i] / 2.0d, y[i]));
        }



        [Fact]
        public void ViennaHostProvider_Scale()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
                x[i] = i;

            var y = new ArraySlice<double>(10);
            provider.Scale(2, x, y);
            provider.Scale(2, x);

            for (int i = 0; i < x.Length; i++)
            {
                Assert.True(MathHelpers.Equality(2 * i, y[i]));
                Assert.True(MathHelpers.Equality(x[i], y[i]));
            }
        }

        [Fact]
        public void ViennaHostProvider_Abs()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
                x[i] = -i;

            var y = new ArraySlice<double>(10);
            provider.Abs(x, y);

            for (int i = 0; i < x.Length; i++)
                Assert.True(MathHelpers.Equality(Math.Abs(x[i]), y[i]));
        }

        [Fact]
        public void ViennaHostProvider_AbsSum()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
                x[i] = -i;

            var y = new ArraySlice<double>(10);
            double asum = provider.Asum(x);

            double cpu = 0;
            for (int i = 0; i < x.Length; i++)
                cpu += Math.Abs(x[i]);

            Assert.True(MathHelpers.Equality(cpu, asum));
        }

        [Fact]
        public void ViennaHostProvider_AbsSumWithRollingArray()
        {
            var provider = new VclHostTensorMath();

            var x = new double[20];
            for (int i = 0; i < x.Length; i++)
                x[i] = -i;

            var results = new double[10];
            for (int i = 0; i < results.Length; i++)
            {
                var y = new ArraySlice<double> (x, i, results.Length);
                results[i] = provider.Asum(x);
            }

            double cpu = 0;
            for (int i = 0; i < 10; i++)
                cpu += Math.Abs(x[i]);

            Assert.All(results, r => MathHelpers.Equality(r, cpu));
        }

        [Fact]
        public void ViennaHostProvider_Dot()
        {
            var provider = new VclHostTensorMath();

            var x = new ArraySlice<double>(10);
            var y = new ArraySlice<double>(10);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = i;
            }

            double result = provider.Dot(x, y);
            double resultEx = provider.Dot(x, 1, y, 1);

            double cpu = 0;
            for (int i = 0; i < x.Length; i++)
                cpu += x[i] * y[i];

            Assert.True(MathHelpers.Equality(cpu, result));
            Assert.True(MathHelpers.Equality(cpu, resultEx));
        }

        [Fact]
        public void ViennaHostProvider_GemmSquare()
        {
            var provider = new VclHostTensorMath();

            int size = 4;

            var x = new ArraySlice<double>(size * size);
            var y = new ArraySlice<double>(size * size);
            for (int i = 0; i < x.Length; i++)
            {
                int ii = i % size;
                int jj = i / size;

                x[i] = i;
                if (ii == jj)
                    y[i] = 1;
            }

            var r = new ArraySlice<double>(size * size);
            provider.Gemm(BlasTranspose.None, BlasTranspose.None, size, size, size, 1.0d, x, y, 0.0d, r);

            for (int i = 0; i < x.Length; i++)
                Assert.True(MathHelpers.Equality(x[i], r[i]));
        }

        [Fact]
        public void ViennaHostProvider_Gemm_SquareWithAdd()
        {
            var provider = new VclHostTensorMath();

            int size = 4;

            var x = new ArraySlice<double>(size * size);
            var y = new ArraySlice<double>(size * size);
            for (int i = 0; i < x.Length; i++)
            {
                int ii = i % size;
                int jj = i / size;

                x[i] = i;
                if (ii == jj)
                    y[i] = 1;
            }

            var r = new ArraySlice<double>(size * size);
            provider.Set(2, r);
            provider.Gemm(BlasTranspose.None, BlasTranspose.None, size, size, size, 1.0d, x, y, 1.0d, r);

            for (int i = 0; i < x.Length; i++)
                Assert.True(MathHelpers.Equality(x[i] + 2, r[i]));
        }


    }
}
