using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace CudaDnn.Impl
{
    /// <summary>
    /// CudnnTensorDescriptorHandle is a pointer to an opaque structure holding the description of a generic 4D dataset. 
    /// CreateTensor() is used to create one instance, and SetParameters() must be used to initialize this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnTensorDescriptorHandle
    {
        public IntPtr Pointer;
    }

    /// <summary>
    /// CudnnConvolutionDescriptorHandle is a pointer to an opaque structure holding the description of a convolution operation. 
    /// CreateConvolution() is used to create one instance, and SetParameters() must be used to initialize this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnConvolutionDescriptorHandle
    {
        public IntPtr Pointer;
    }

    /// <summary>
    /// CudnnPoolingDescriptorHandle is a pointer to an opaque structure holding the description of a pooling operation. 
    /// CreatePooling() is used to create one instance, and SetParameters() must be used to initialize this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnPoolingDescriptorHandle
    {
        public IntPtr Pointer;
    }

    /// <summary>
    /// CudnnFilterDescriptorHandle is a pointer to an opaque structure holding the description of a filter dataset. 
    /// CreateFilter() is used to create one instance, and SetParameters() must be used to initialize this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnFilterDescriptorHandle
    {
        public IntPtr Pointer;
    }
}
