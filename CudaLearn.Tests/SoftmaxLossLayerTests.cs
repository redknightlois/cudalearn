using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{

    public class SoftmaxLossLayerTests : CpuLayerTests
    {
        private readonly Tensor bottom = new Tensor(10, 5, 1, 1);
        private readonly Tensor labels = new Tensor(10, 1, 1, 1);

        public SoftmaxLossLayerTests()
        {
            var filler = new GaussianFiller();
            filler.Fill(bottom);

            Random rnd = new Random(1000);

            using (var labelsCpu = labels.OnCpu())
            {
                var labelData = labelsCpu.Data;
                for (int i = 0; i < labelData.Count; i++)
                    labelData[i] = rnd.Next(5);
            }


        }

        [Fact]
        public void SoftmaxLayer_BackwardGradient()
        {
            var layer = new SoftmaxLayer();
            layer.Setup(bottom, labels);

            var checker = new GradientChecker(1e-2f, 1e-2f);
            checker.CheckSingle(layer, bottom, labels, 0, -1, -1);
        }
    }
}
