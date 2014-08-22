using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public struct Size
    {
        public int Width;
        public int Height;
        public int Depth;

        public Size(int height = 0, int width = 0, int depth = 0)
        {
            this.Height = height;
            this.Width = width;
            this.Depth = depth;
        }

        public Size(Size other)
        {
            this.Height = other.Height;
            this.Width = other.Width;
            this.Depth = other.Depth;
        }

        public override string ToString()
        {
            return base.ToString() + "(" + Height + "," + Width + "," + Depth + ")";
        }
    }
}
