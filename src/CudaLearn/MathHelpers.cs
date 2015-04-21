using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class MathHelpers
    {
        public const double Epsilon = 0.0001f;

        public static bool Equality(double a, double b, double tolerance = Epsilon)
        {
            return Math.Abs(a - b) <= tolerance;
        }
    }
}
