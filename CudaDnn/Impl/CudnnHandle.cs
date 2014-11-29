using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace CudaDnn.Impl
{
    /// <summary>
    /// CudnnHandle is a pointer to an opaque structure holding the cuDNN library context. 
    /// The cuDNN library context must be created using Create() and the returned handle must be passed to all subsequent library function calls. 
    /// The context should be destroyed at the end using Dispose(). The context is associated with only one GPU device, the current device at the time of the call 
    /// to Create(). However multiple contexts can be created on the same GPU device.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnHandle
    {
        public IntPtr Pointer;
    }

}
