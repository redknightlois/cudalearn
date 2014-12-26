using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    /// <summary>
    /// CudnnType is an enumerated type indicating the data type to which a tensor descriptor or filter descriptor refers.
    /// </summary>
    public enum CudnnType
    {
        /// <summary>
        /// The data is 32-bit single-precision floating point (float).
        /// </summary>
        Float = 0,
        /// <summary>
        /// The data is 64-bit double-precision floating point (double).
        /// </summary>
        Double = 1,
    }
}
