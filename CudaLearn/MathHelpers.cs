using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class MathHelpers
    {
        public const float Epsilon = 0.00001f;

        public static bool Equality(float a, float b, float tolerance = Epsilon)
        {
            return Math.Abs(a - b) <= tolerance;
        }

        public static bool Equality(double a, double b, double tolerance = Epsilon)
        {
            return Math.Abs(a - b) <= tolerance;
        }
    }
}
