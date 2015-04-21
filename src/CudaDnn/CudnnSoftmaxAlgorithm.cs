using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    /// <summary>
    /// CudnnSoftmaxAlgorithm is used to select an implementation of the softmax function used in Forward() and Backward().
    /// </summary>
    public enum CudnnSoftmaxAlgorithm
    {
        /// <summary>
        /// This implementation applies the straightforward softmax operation.
        /// </summary>
        Fast = 0,
        /// <summary>
        /// This implementation subtract max from every point to avoid any potential overflow.
        /// </summary>
        Accurate = 1
    }
}
