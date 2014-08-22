using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Extensions;

namespace CudaLearn.Tests
{

    public class MaxPoolingLayerTests
    {
        private readonly Blob bottom = new Blob(2, 3, 6, 5);
        private readonly Blob top = new Blob();

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
            const int num = 2;
            const int channels = 2;
            var bottom = new Blob(num, channels, 3, 5);

            var topList = new Blob[topLayer];
            for (int i = 0; i < topLayer; i++)
                topList[i] = new Blob();

            // Input: 2x 2 channels of:
            //     [1 2 5 2 3]
            //     [9 4 1 4 8]
            //     [1 2 5 2 3]

            var bottomData = bottom.Data;
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

            var layer = new MaxPoolingLayer(2, 1, 0);
            layer.Setup(new[] { bottom }, topList);

            foreach ( var top in topList)
            {
                Assert.Equal(num, top.Num);
                Assert.Equal(channels, top.Channels);
                Assert.Equal(2, top.Height);
                Assert.Equal(4, top.Width);
            }

            layer.Forward(new[] { bottom }, topList);

            // Expected output: 2x 2 channels of:
            //     [9 5 5 8]
            //     [9 5 5 8]
            for (int i = 0; i < 8 * num * channels; i += 8)
            {
                var topData = topList[0].Data;
                Assert.Equal(9,topData[i + 0]);
                Assert.Equal(5,topData[i + 1]);
                Assert.Equal(5,topData[i + 2]);
                Assert.Equal(8,topData[i + 3]);
                Assert.Equal(9,topData[i + 4]);
                Assert.Equal(5,topData[i + 5]);
                Assert.Equal(5,topData[i + 6]);
                Assert.Equal(8,topData[i + 7]);
            }

            if ( topList.Length > 1 )
            {
                // Expected mask output: 2x 2 channels of:
                //     [5  2  2 9]
                //     [5 12 12 9]
                for (int i = 0; i < 8 * num * channels; i += 8)
                {
                    var topData = topList[1].Data;
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

        [Theory]
        [InlineData(1), InlineData(2)]
        public void MaxPoolingLayer_BackwardsRectangularWithSquareKernelConstant(int topLayer)
        {
            const int num = 2;
            const int channels = 2;
            var bottom = new Blob(num, channels, 3, 5);

            var topList = new Blob[topLayer];
            for (int i = 0; i < topLayer; i++)
                topList[i] = new Blob();

            var filler = new ConstantFiller(2);
            filler.Fill(bottom);

            // Input: 2x 2 channels of:
            //     [2 2 2 2 2]
            //     [2 2 2 2 2]
            //     [2 2 2 2 2]

            var layer = new MaxPoolingLayer(2, 1, 0);
            layer.Setup(new[] { bottom }, topList);
            layer.Forward(new[] { bottom }, topList);

            // Input: 2x 2 channels of:
            //     [1 0 0 0]
            //     [0 0 0 1]

            for (int i = 0; i < 8 * num * channels; i += 8)
            {
                var topDiff = topList[0].Diff;
                topDiff[i + 0] = 1;
                topDiff[i + 7] = 1;
            }

            // Input: 2x 2 channels of:
            //     [1 1 0 0 0]
            //     [1 1 0 1 1]
            //     [0 0 0 1 1]

            layer.Backward(topList, new[] { true }, new[] { bottom });

            for (int i = 0; i < 15 * num * channels; i += 15)
            {
                var bottomDiff = bottom.Diff;
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


        [Theory]
        [InlineData(1), InlineData(2)]
        public void MaxPoolingLayer_BackwardsRectangularWithSquareKernelConstantStrips(int topLayer)
        {
            const int num = 2;
            const int channels = 2;
            var bottom = new Blob(num, channels, 3, 5);

            var topList = new Blob[topLayer];
            for (int i = 0; i < topLayer; i++)
                topList[i] = new Blob();

            var filler = new ConstantFiller(2);
            filler.Fill(bottom);

            // Input: 2x 2 channels of:
            //     [2 1 2 1 2]
            //     [2 1 2 1 2]
            //     [2 1 2 1 2]

            for (int i = 0; i < 15 * num * channels; i += 15)
            {
                var bottomData = bottom.Data;

                bottomData[i + 1] = bottomData[i + 3] = 1;
                bottomData[i + 6] = bottomData[i + 8] = 1;
                bottomData[i + 11] = bottomData[i + 13] = 1;
            }

            var layer = new MaxPoolingLayer(2, 1, 0);
            layer.Setup(new[] { bottom }, topList);
            layer.Forward(new[] { bottom }, topList);

            // Input: 2x 2 channels of:
            //     [1 0 0 0]
            //     [0 0 0 1]

            for (int i = 0; i < 8 * num * channels; i += 8)
            {
                var topDiff = topList[0].Diff;
                topDiff[i + 0] = 1;
                topDiff[i + 7] = 1;
            }

            // Input: 2x 2 channels of:
            //     [1 0 0 0 0]
            //     [1 0 0 0 1]
            //     [0 0 0 0 1]

            layer.Backward(topList, new[] { true }, new[] { bottom });

            for (int i = 0; i < 15 * num * channels; i += 15)
            {
                var bottomDiff = bottom.Diff;
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

        [Theory]
        [InlineData(1), InlineData(2)]
        public void MaxPoolingLayer_BackwardsRectangularWithSquareKernel(int topLayer)
        {
            const int num = 2;
            const int channels = 2;
            var bottom = new Blob(num, channels, 3, 5);

            var topList = new Blob[topLayer];
            for (int i = 0; i < topLayer; i++)
                topList[i] = new Blob();

            var filler = new ConstantFiller(2);
            filler.Fill(bottom);

            // Input: 2x 2 channels of:
            //     [2 2 2 2 2]
            //     [2 2 2 2 2]
            //     [2 2 2 2 2]

            var layer = new MaxPoolingLayer(2, 1, 0);
            layer.Setup(new[] { bottom }, topList);
            layer.Forward(new[] { bottom }, topList);

            // Input: 2x 2 channels of:
            //     [1 1 1 1]
            //     [0 0 0 0]

            for (int i = 0; i < 8 * num * channels; i += 8)
            {
                var topDiff = topList[0].Diff;
                topDiff[i + 0] = 1;
                topDiff[i + 1] = 1;
                topDiff[i + 2] = 1;
                topDiff[i + 3] = 1;
            }

            // Input: 2x 2 channels of:
            //     [1 2 2 2 1]
            //     [1 2 2 2 1]
            //     [0 0 0 0 0]

            layer.Backward(topList, new[] { true }, new[] { bottom });

            for (int i = 0; i < 15 * num * channels; i += 15)
            {
                var bottomDiff = bottom.Diff;
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
