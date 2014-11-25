using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    public enum CudnnSoftmaxMode
    {
        /// <summary>
        /// compute the softmax over all C, H, W for each N 
        /// </summary>
        Instance = 0,
        /// <summary>
        /// compute the softmax over all C for each H, W, N 
        /// </summary>
        Channel = 1
    }
}
