using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    /// <summary>
    /// CudnnSoftmaxMode is used to select over which data the Forward() and Backward() are computing their results.
    /// </summary>
    public enum CudnnSoftmaxMode
    {
        /// <summary>
        /// The softmax operation is computed per image (N) across the dimensions C,H,W.
        /// </summary>
        Instance = 0,
        /// <summary>
        /// The softmax operation is computed per spatial location (H,W) per image (N) across the dimension C.
        /// </summary>
        Channel = 1
    }
}
