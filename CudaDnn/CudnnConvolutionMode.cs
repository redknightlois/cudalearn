using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    /// <summary>
    /// CudnnConvolutionMode is an enumerated type used by SetParameters() to configure a convolution descriptor. 
    /// The filter used for the convolution can be applied in two different ways, corresponding mathematically to a convolution or to a cross-correlation. 
    /// (A cross-correlation is equivalent to a convolution with its filter rotated by 180 degrees.)
    /// </summary>
    public enum CudnnConvolutionMode
    {
        /// <summary>
        /// In this mode, a convolution operation will be done when applying the filter to the images
        /// </summary>
        Convolution = 0,
        /// <summary>
        /// In this mode, a cross-correlation operation will be done when applying the filter to the images.
        /// </summary>
        CrossCorrelation = 1,
    }
}
