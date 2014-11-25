using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace CudaDnn
{
    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnTensorDescriptor
    {
        public IntPtr Pointer;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnConvolutionDescriptor
    {
        public IntPtr Pointer;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnPoolingDescriptor
    {
        public IntPtr Pointer;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CudnnFilterDescriptor
    {
        public IntPtr Pointer;
    }
}
