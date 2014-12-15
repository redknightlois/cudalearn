using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace System.Collections.Generic
{
    public static class ListExtensions
    {
        public static List<T> RepeatedDefault<T>(this List<T> list, int count)
        {
            Contract.Requires(list != null);
            Contract.Requires(count > 0);
            Contract.Ensures(Contract.Result<List<T>>() != null);

            return Repeated(list, count, default(T));
        }

        public static List<T> Repeated<T>(this List<T> list, int count, T value)
        {
            Contract.Requires(list != null);
            Contract.Requires(count > 0);
            Contract.Ensures(Contract.Result<List<T>>() != null);            

            Guard.That(() => count).IsPositive();
            Guard.That(() => list).IsNotNull();            

            list.AddRange(Enumerable.Repeat(value, count));
            return list;
        }
    }
}
