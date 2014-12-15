using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class ComparableValidatorExtensions
    {
        public static IArg<T> IsGreaterOrEqualThan<T>(this IArg<T> arg, T param) where T : IComparable
        {
            Contract.Requires(arg != null);
            Contract.Requires(param != null);
            Contract.Ensures(Contract.Result<IArg<T>>() != null);

            return IsGreaterOrEqualThan(arg, () => param);
        }

        public static IArg<T> IsGreaterOrEqualThan<T>(this IArg<T> arg, Func<T> param) where T : IComparable
        {
            Contract.Requires(arg != null);
            Contract.Requires(param != null);
            Contract.Ensures(Contract.Result<IArg<T>>() != null);

            if (arg.Value.CompareTo(param()) < 0)
            {
                arg.Message.SetArgumentOutRange();
            }

            return arg;
        }

        public static IArg<T> IsLessOrEqualThan<T>(this IArg<T> arg, T param) where T : IComparable
        {
            Contract.Requires(arg != null);
            Contract.Requires(param != null);
            Contract.Ensures(Contract.Result<IArg<T>>() != null);

            return IsLessOrEqualThan(arg, () => param);
        }

        public static IArg<T> IsLessOrEqualThan<T>(this IArg<T> arg, Func<T> param) where T : IComparable
        {
            Contract.Requires(arg != null);
            Contract.Requires(param != null);
            Contract.Ensures(Contract.Result<IArg<T>>() != null);

            if (arg.Value.CompareTo(param()) > 0)
            {
                arg.Message.SetArgumentOutRange();
            }

            return arg;
        }
    }
}
