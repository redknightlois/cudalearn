using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{

    public class MaxPoolingLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(2, 3, 6, 5);
        private readonly Tensor top = new Tensor();

        public MaxPoolingLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);
        }

        [Fact]
        public void MaxPoolingLayer_Setup()
        {
            var layer = new MaxPoolingLayer(3, 2);
            layer.Setup(bottom, top);

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(3, top.Height);
            Assert.Equal(2, top.Width);
        }

        [Fact]
        public void MaxPoolingLayer_SetupPadded()
        {
            var layer = new MaxPoolingLayer(3, 2, 1);
            layer.Setup(bottom, top);            

            Assert.Equal(bottom.Num, top.Num);
            Assert.Equal(bottom.Channels, top.Channels);
            Assert.Equal(4, top.Height);
            Assert.Equal(3, top.Width);
        }

        [Theory]
        [InlineData(1), InlineData(2)]
        public void MaxPoolingLayer_ForwardRectangularWithSquareKernel(int topLayer)
        {
            Contract.Requires(topLayer > 0);

            const int num = 2;
            const int channels = 2;
            var bottom = new Tensor(num, channels, 3, 5);

            var topList = new Tensor[topLayer];
            for (int i = 0; i < topLayer; i++)
                topList[i] = new Tensor();

            // Input: 2x 2 channels of:
            //     [1 2 5 2 3]
            //     [9 4 1 4 8]
            //     [1 2 5 2 3]

            using (var bottomCpu = bottom.OnCpu())
            {
                var bottomData = bottomCpu.Data;
                for (int i = 0; i < 15 * num * channels; i += 15)
                {
                    bottomData[i + 0] = 1;
                    bottomData[i + 1] = 2;
                    bottomData[i + 2] = 5;
                    bottomData[i + 3] = 2;
                    bottomData[i + 4] = 3;
                    bottomData[i + 5] = 9;
                    bottomData[i + 6] = 4;
                    bottomData[i + 7] = 1;
                    bottomData[i + 8] = 4;
                    bottomData[i + 9] = 8;
                    bottomData[i + 10] = 1;
                    bottomData[i + 11] = 2;
                    bottomData[i + 12] = 5;
                    bottomData[i + 13] = 2;
                    bottomData[i + 14] = 3;
                }
            }

            var layer = new MaxPoolingLayer(2, 1, 0);
            layer.Setup(new TensorCollection { bottom }, topList);

            foreach ( var top in topList)
            {
                Assert.Equal(num, top.Num);
                Assert.Equal(channels, top.Channels);
                Assert.Equal(2, top.Height);
                Assert.Equal(4, top.Width);
            }

            layer.Forward(new TensorCollection { bottom }, topList);

            // Expected output: 2x 2 channels of:
            //     [9 5 5 8]
            //     [9 5 5 8]
            for (int i = 0; i < 8 * num * channels; i += 8)
            {
                using (var topCpu = topList[0].OnCpu())
                {
                    var topData = topCpu.Data;
                    Assert.Equal(9, topData[i + 0]);
                    Assert.Equal(5, topData[i + 1]);
                    Assert.Equal(5, topData[i + 2]);
                    Assert.Equal(8, topData[i + 3]);
                    Assert.Equal(9, topData[i + 4]);
                    Assert.Equal(5, topData[i + 5]);
                    Assert.Equal(5, topData[i + 6]);
                    Assert.Equal(8, topData[i + 7]);
                }
            }

            if ( topList.Length > 1 )
            {
                // Expected mask output: 2x 2 channels of:
                //     [5  2  2 9]
                //     [5 12 12 9]
                for (int i = 0; i < 8 * num * channels; i += 8)
                {
                    using (var topCpu = topList[1].OnCpu())
                    {
                        var topData = topCpu.Data;
                        Assert.Equal(5, topData[i + 0]);
                        Assert.Equal(2, topData[i + 1]);
                        Assert.Equal(2, topData[i + 2]);
                        Assert.Equal(9, topData[i + 3]);
                        Assert.Equal(5, topData[i + 4]);
                        Assert.Equal(12, topData[i + 5]);
                        Assert.Equal(12, topData[i + 6]);
                        Assert.Equal(9, topData[i + 7]);
                    }
                }
            }
        }

        [Theory]
        [InlineData(1), InlineData(2)]
        public void MaxPoolingLayer_BackwardsRectangularWithSquareKernelConstant(int topLayer)
        {
            Contract.Requires(topLayer > 0);

            const int num = 2;
            const int channels = 2;
            var bottom = new Tensor(num, channels, 3, 5);

            var topList = new Tensor[topLayer];
            for (int i = 0; i < topLayer; i++)
                topList[i] = new Tensor();

            var filler = new ConstantFiller(2);
            filler.Fill(bottom);

            // Input: 2x 2 channels of:
            //     [2 2 2 2 2]
            //     [2 2 2 2 2]
            //     [2 2 2 2 2]

            var layer = new MaxPoolingLayer(2, 1, 0);
            layer.Setup(new TensorCollection { bottom }, topList);
            layer.Forward(new TensorCollection { bottom }, topList);

            // Input: 2x 2 channels of:
            //     [1 0 0 0]
            //     [0 0 0 1]

            using (var topDiffCpu = topList[0].OnCpu())
            {
                for (int i = 0; i < 8 * num * channels; i += 8)
                {
                    var topDiff = topDiffCpu.Diff;
                    topDiff[i + 0] = 1;
                    topDiff[i + 7] = 1;
                }
            }


            // Input: 2x 2 channels of:
            //     [1 1 0 0 0]
            //     [1 1 0 1 1]
            //     [0 0 0 1 1]

            layer.Backward(topList, new[] { true }, new TensorCollection { bottom });

            using (var bottomCpu = bottom.OnCpu())
            {
                var bottomDiff = bottomCpu.Diff;
                for (int i = 0; i < 15 * num * channels; i += 15)
                {
                    Assert.Equal(1, bottomDiff[i + 0]);
                    Assert.Equal(1, bottomDiff[i + 1]);
                    Assert.Equal(0, bottomDiff[i + 2]);
                    Assert.Equal(0, bottomDiff[i + 3]);
                    Assert.Equal(0, bottomDiff[i + 4]);
                    Assert.Equal(1, bottomDiff[i + 5]);
                    Assert.Equal(1, bottomDiff[i + 6]);
                    Assert.Equal(0, bottomDiff[i + 7]);
                    Assert.Equal(1, bottomDiff[i + 8]);
                    Assert.Equal(1, bottomDiff[i + 9]);
                    Assert.Equal(0, bottomDiff[i + 10]);
                    Assert.Equal(0, bottomDiff[i + 11]);
                    Assert.Equal(0, bottomDiff[i + 12]);
                    Assert.Equal(1, bottomDiff[i + 13]);
                    Assert.Equal(1, bottomDiff[i + 14]);
                }
            }
        }


        [Theory]
        [InlineData(1), InlineData(2)]
        public void MaxPoolingLayer_BackwardsRectangularWithSquareKernelConstantStrips(int topLayer)
        {
            Contract.Requires(topLayer > 0);

            const int num = 2;
            const int channels = 2;
            var bottom = new Tensor(num, channels, 3, 5);

            var topList = new Tensor[topLayer];
            for (int i = 0; i < topLayer; i++)
                topList[i] = new Tensor();

            var filler = new ConstantFiller(2);
            filler.Fill(bottom);

            // Input: 2x 2 channels of:
            //     [2 1 2 1 2]
            //     [2 1 2 1 2]
            //     [2 1 2 1 2]

            using (var bottomCpu = bottom.OnCpu())
            {
                for (int i = 0; i < 15 * num * channels; i += 15)
                {
                    var bottomData = bottomCpu.Data;

                    bottomData[i + 1] = bottomData[i + 3] = 1;
                    bottomData[i + 6] = bottomData[i + 8] = 1;
                    bottomData[i + 11] = bottomData[i + 13] = 1;
                }
            } 

            var layer = new MaxPoolingLayer(2, 1, 0);
            layer.Setup(new TensorCollection { bottom }, topList);
            layer.Forward(new TensorCollection { bottom }, topList);

            // Input: 2x 2 channels of:
            //     [1 0 0 0]
            //     [0 0 0 1]

            using (var topCpu = topList[0].OnCpu())
            {
                for (int i = 0; i < 8 * num * channels; i += 8)
                {
                    var topDiff = topCpu.Diff;
                    topDiff[i + 0] = 1;
                    topDiff[i + 7] = 1;
                }
            }

            // Input: 2x 2 channels of:
            //     [1 0 0 0 0]
            //     [1 0 0 0 1]
            //     [0 0 0 0 1]

            layer.Backward(topList, new[] { true }, new TensorCollection { bottom });

            using (var bottomCpu = bottom.OnCpu())
            {
                var bottomDiff = bottomCpu.Diff;
                for (int i = 0; i < 15 * num * channels; i += 15)
                {
                    Assert.Equal(1, bottomDiff[i + 0]);
                    Assert.Equal(0, bottomDiff[i + 1]);
                    Assert.Equal(0, bottomDiff[i + 2]);
                    Assert.Equal(0, bottomDiff[i + 3]);
                    Assert.Equal(0, bottomDiff[i + 4]);
                    Assert.Equal(1, bottomDiff[i + 5]);
                    Assert.Equal(0, bottomDiff[i + 6]);
                    Assert.Equal(0, bottomDiff[i + 7]);
                    Assert.Equal(0, bottomDiff[i + 8]);
                    Assert.Equal(1, bottomDiff[i + 9]);
                    Assert.Equal(0, bottomDiff[i + 10]);
                    Assert.Equal(0, bottomDiff[i + 11]);
                    Assert.Equal(0, bottomDiff[i + 12]);
                    Assert.Equal(0, bottomDiff[i + 13]);
                    Assert.Equal(1, bottomDiff[i + 14]);
                }
            }
        }

        [Theory]
        [InlineData(1), InlineData(2)]
        public void MaxPoolingLayer_BackwardsRectangularWithSquareKernel(int topLayer)
        {
            Contract.Requires(topLayer > 0);

            const int num = 2;
            const int channels = 2;
            var bottom = new Tensor(num, channels, 3, 5);

            var topList = new Tensor[topLayer];
            for (int i = 0; i < topLayer; i++)
                topList[i] = new Tensor();

            var filler = new ConstantFiller(2);
            filler.Fill(bottom);

            // Input: 2x 2 channels of:
            //     [2 2 2 2 2]
            //     [2 2 2 2 2]
            //     [2 2 2 2 2]

            var layer = new MaxPoolingLayer(2, 1, 0);
            layer.Setup(new TensorCollection { bottom }, topList);
            layer.Forward(new TensorCollection { bottom }, topList);

            // Input: 2x 2 channels of:
            //     [1 1 1 1]
            //     [0 0 0 0]


            using (var topCpu = topList[0].OnCpu())
            {
                var topDiff = topCpu.Diff;
                for (int i = 0; i < 8 * num * channels; i += 8)
                {

                    topDiff[i + 0] = 1;
                    topDiff[i + 1] = 1;
                    topDiff[i + 2] = 1;
                    topDiff[i + 3] = 1;
                }
            }

            // Input: 2x 2 channels of:
            //     [1 2 2 2 1]
            //     [1 2 2 2 1]
            //     [0 0 0 0 0]

            layer.Backward(topList, new[] { true }, new TensorCollection { bottom });

            using (var bottomCpu = bottom.OnCpu())
            {
                var bottomDiff = bottomCpu.Diff;
                for (int i = 0; i < 15 * num * channels; i += 15)
                {
                    Assert.Equal(1, bottomDiff[i + 0]);
                    Assert.Equal(2, bottomDiff[i + 1]);
                    Assert.Equal(2, bottomDiff[i + 2]);
                    Assert.Equal(2, bottomDiff[i + 3]);
                    Assert.Equal(1, bottomDiff[i + 4]);
                    Assert.Equal(1, bottomDiff[i + 5]);
                    Assert.Equal(2, bottomDiff[i + 6]);
                    Assert.Equal(2, bottomDiff[i + 7]);
                    Assert.Equal(2, bottomDiff[i + 8]);
                    Assert.Equal(1, bottomDiff[i + 9]);
                    Assert.Equal(0, bottomDiff[i + 10]);
                    Assert.Equal(0, bottomDiff[i + 11]);
                    Assert.Equal(0, bottomDiff[i + 12]);
                    Assert.Equal(0, bottomDiff[i + 13]);
                    Assert.Equal(0, bottomDiff[i + 14]);
                }
            }
        }
    }
}
