using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class IntegerValidatorExtensions
    {
        public static IArg<long> IsNatural(this IArg<long> arg)
        {
            Contract.Requires(arg != null);
            Contract.Ensures(arg != null);

            if (arg.Value < 0)
                arg.Message.Set("Is not natural. Zero or greater.");

            return arg;
        }

        public static IArg<int> IsNatural(this IArg<int> arg)
        {
            Contract.Requires(arg != null);
            Contract.Ensures(arg != null);

            if (arg.Value < 0)
                arg.Message.Set("Is not natural. Zero or greater.");

            return arg;
        }
    }
}
