using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn.Examples.Rbm
{

    public class MnistDatabase
    {
        public int Samples { get; private set; }
        public int SampleSize { get; private set; }

        public IEnumerable<Matrix<float>> Batches(int batchSize)
        {
            throw new NotImplementedException();
        }
    }
}
