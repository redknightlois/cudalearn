using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    public enum CudnnConvolutionPath
    {
        /// <summary>
        /// Tensor Convolution function
        /// </summary>
        Forward = 0,
        /// <summary>
        /// Weight Gradient update function
        /// </summary>
        WeightsUpdate = 1,
        /// <summary>
        /// Data Gradient update function
        /// </summary>
        DataUpdate = 2,
    }
}
