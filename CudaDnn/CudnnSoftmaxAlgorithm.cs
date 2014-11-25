using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    public enum CudnnSoftmaxAlgorithm
    {
        /// <summary>
        /// straightforward implementation
        /// </summary>
        Fast = 0,
        /// <summary>
        /// subtract max from every point to avoid overflow
        /// </summary>
        Accurate = 1
    }
}
