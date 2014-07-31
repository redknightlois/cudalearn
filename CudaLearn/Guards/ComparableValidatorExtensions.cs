using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class ComparableValidatorExtensions
    {
        public static IArg<T> IsGreaterOrEqualThan<T>(this IArg<T> arg, T param) where T : IComparable
        {
            return IsGreaterOrEqualThan(arg, () => param);
        }

        public static IArg<T> IsGreaterOrEqualThan<T>(this IArg<T> arg, Func<T> param) where T : IComparable
        {
            if (arg.Value.CompareTo(param()) < 0)
            {
                arg.Message.SetArgumentOutRange();
            }

            return arg;
        }

        public static IArg<T> IsLessOrEqualThan<T>(this IArg<T> arg, T param) where T : IComparable
        {
            return IsLessOrEqualThan(arg, () => param);
        }

        public static IArg<T> IsLessOrEqualThan<T>(this IArg<T> arg, Func<T> param) where T : IComparable
        {
            if (arg.Value.CompareTo(param()) > 0)
            {
                arg.Message.SetArgumentOutRange();
            }

            return arg;
        }
    }
}
