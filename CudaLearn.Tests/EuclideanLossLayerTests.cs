using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class EuclideanLossLayerTests
    {
        private readonly Blob blobBottomData;
        private readonly Blob blobBottomLabel;

        private readonly IList<Blob> bottom = new List<Blob>();
        private readonly IList<Blob> top = new List<Blob>();

        public EuclideanLossLayerTests()
        {             
            var filler = new GaussianFiller();

            blobBottomData = new Blob(10, 5, 1, 1);
            filler.Fill(blobBottomData);
            bottom.Add(blobBottomData);

            blobBottomLabel = new Blob(10, 5, 1, 1);
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
