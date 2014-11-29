using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    /// <summary>
    /// CudnnConvolutionPath is an enumerated type used by the helper routine GetOutputDimensions() to select the results to output.
    /// </summary>
    public enum CudnnConvolutionPath
    {
        /// <summary>
        /// Tensor Convolution function
        /// GetOutputDimensions() will return dimensions related to the output tensor of the forward convolution.
        /// </summary>
        Forward = 0,
        /// <summary>
        /// Weight Gradient update function
        /// GetOutputDimensions() will return the dimensions of the output filter produced while computing the gradients, which is part of the backward convolution.
        /// </summary>
        WeightsUpdate = 1,
        /// <summary>
        /// Data Gradient update function
        /// GetOutputDimensions() will return the dimensions of the output tensor produced while computing the gradients, which is part of the backward convolution.
        /// </summary>
        DataUpdate = 2,
    }
}
