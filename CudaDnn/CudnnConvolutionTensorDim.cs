using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    public class CudnnConvolutionTensorDim
    {
        public readonly CudnnConvolutionPath Path;

        public readonly int Num;
        public readonly int Channels;
        public readonly int Height;
        public readonly int Width;

        public CudnnConvolutionTensorDim( CudnnConvolutionPath path, int n, int c, int h, int w)
        {
            this.Path = path;

            this.Num = n;
            this.Channels = c;
            this.Height = h;
            this.Width = w;
        }
    }
}
