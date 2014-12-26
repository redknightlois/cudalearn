using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    public enum CudnnStatus
    {
        /// <summary>
        /// The operation completed successfully.
        /// </summary>
        Success = 0,
        /// <summary>
        /// The cuDNN library was not initialized properly. This error is usually returned when a call to cudnnCreate() fails or when cudnnCreate()
        /// has not been called prior to calling another cuDNN routine. In the former case, it is usually due to an error in the CUDA Runtime API called by
        /// cudnnCreate() or by an error in the hardware setup.
        /// </summary>
        NotInitialized = 1,
        /// <summary>
        /// Resource allocation failed inside the cuDNN library. This is usually caused by an internal cudaMalloc() failure.
        /// To correct: prior to the function call, deallocate previously allocated memory as much as possible.
        /// </summary>
        AllocationFailed = 2,
        /// <summary>
        /// An incorrect value or parameter was passed to the function. To correct: ensure that all the parameters being passed have valid values.
        /// </summary>
        BadParameter = 3,
        /// <summary>
        /// An internal cuDNN operation failed.
        /// </summary>
        InternalError = 4,
        InvalidValue = 5,
        /// <summary>
        /// The function requires a feature absent from the current GPU device. Note that cuDNN only supports devices with compute capabilities greater
        /// than or equal to 3.0. To correct: compile and run the application on a device with appropriate compute capability.
        /// </summary>
        ArchitectureMismatch = 6,
        /// <summary>
        /// An access to GPU memory space failed, which is usually caused by a failure to bind a texture. 
        /// To correct: prior to the function call, unbind any previously bound textures. Otherwise, this may indicate an internal error/bug
        /// in the library.
        /// </summary>
        MappingError = 7,
        /// <summary>
        /// The GPU program failed to execute. This is usually caused by a failure to launch some cuDNN kernel on the GPU, which can occur for multiple reasons.
        /// To correct: check that the hardware, an appropriate version of the driver, and the cuDNN library are correctly installed.
        /// Otherwise, this may indicate a internal error/bug in the library
        /// </summary>
        ExecutionFailed = 8,
        /// <summary>
        /// The functionality requested is not presently supported by cuDNN.
        /// </summary>
        NotSupported = 9,
        /// <summary>
        /// The functionality requested requires some license and an error was detected when trying to check the current licensing. 
        /// This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE
        /// is not set properly.
        /// </summary>
        LicenseError = 10,
    }
}
