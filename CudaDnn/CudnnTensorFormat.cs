using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    public enum CudnnTensorFormat
    {
        MajorRow = 0, // NCHW - row major (wStride = 1, hStride = w)        
        Interleaved = 1, // NHWC - feature maps interleaved ( cStride = 1 )

        NCHW = 0,
        NHWC = 1,
    }
}
