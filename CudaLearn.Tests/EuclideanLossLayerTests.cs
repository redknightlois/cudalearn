using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    //TODO Implement a tests that will use a known loss and check accordingly. I am not entirely sure that the current gradient based implementation is correct.
    public class EuclideanLossLayerTests : CpuLayerTests
    {
        private readonly Tensor blobBottomData;
        private readonly Tensor blobBottomLabel;

        private readonly TensorCollection bottom = new TensorCollection();
        private readonly TensorCollection top = new TensorCollection();

        public EuclideanLossLayerTests()
        {             
            var filler = new GaussianFiller();

            blobBottomData = new Tensor(10, 5, 1, 1);
            filler.Fill(blobBottomData);
            bottom.Add(blobBottomData);

            blobBottomLabel = new Tensor(10, 5, 1, 1);
            filler.Fill(blobBottomLabel);
            bottom.Add(blobBottomLabel);
        }


        [Fact]
        public void EuclideanLossLayer_BackwardGradient()
        {
            var layer = new EuclideanLossLayer();
            layer.Setup(bottom, top);

            var checker = new GradientChecker(1e-6f, 4e-1f);
            checker.CheckSingle(layer, bottom, top, -1, -1, -1);
        }

    }
}
