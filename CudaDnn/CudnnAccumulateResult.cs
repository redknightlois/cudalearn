using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    /// <summary>
    /// Accumulate the result of the operation into output buffer (or overwrite)
    /// </summary>
    public enum CudnnAccumulateResult
    {
        /// <summary>
        /// Evaluate O += I * F 
        /// </summary>
        Accumulate,
        /// <summary>
        /// Evaluate O = I * F
        /// </summary>
        DoNotAccumulate,
    }
}
