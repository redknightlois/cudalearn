using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace CudaDnn.Impl
{
    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnTensorDescriptorHandle
    {
        public IntPtr Pointer;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnConvolutionDescriptorHandle
    {
        public IntPtr Pointer;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnPoolingDescriptorHandle
    {
        public IntPtr Pointer;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnFilterDescriptorHandle
    {
        public IntPtr Pointer;
    }
}
