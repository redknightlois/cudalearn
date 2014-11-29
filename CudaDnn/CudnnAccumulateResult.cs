using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    /// <summary>
    /// CudnnAccumulateResult is an enumerated type used by Forward(), BackwardFilter() and BackwardData() to specify 
    /// whether those routines accumulate their results with the output tensor or simply write them to it, overwriting the previous value.
    /// </summary>
    public enum CudnnAccumulateResult
    {
        /// <summary>
        /// Evaluate O += I * F 
        /// The results are accumulated with (added to the previous value of) the output tensor.
        /// </summary>
        Accumulate,
        /// <summary>
        /// Evaluate O = I * F
        /// The results overwrite the output tensor.
        /// </summary>
        DoNotAccumulate,
    }
}
