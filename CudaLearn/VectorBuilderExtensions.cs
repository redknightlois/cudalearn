using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class VectorBuilderExtensions
    {
        public static Vector<T> SameAs<T>(this VectorBuilder<T> builder, Vector<T> example, Func<T> f) where T : struct, global::System.IEquatable<T>, global::System.IFormattable
        {
            Contract.Requires(builder != null);
            Contract.Requires(example != null);
            Contract.Requires(f != null);            

            var result = builder.SameAs(example);
            Contract.Assume(result != null);

            result.MapInplace(x => f());
            return result;
        }

    }
}
