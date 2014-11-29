using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    /// <summary>
    /// CudnnActivationMode is an enumerated type used to select the neuron activation function used in Forward() and Backward().
    /// </summary>
    public enum CudnnActivationMode
    {
        /// <summary>
        /// Selects the sigmoid function.
        /// </summary>
        Sigmoid,
        /// <summary>
        /// Selects the rectified linear function.
        /// </summary>
        Relu,
        /// <summary>
        /// Selects the hyperbolic tangent function.
        /// </summary>
        Tanh,
    }
}
