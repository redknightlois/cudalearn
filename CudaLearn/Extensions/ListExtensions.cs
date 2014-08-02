using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace System.Collections.Generic
{
    public static class ListExtensions
    {
        public static List<T> RepeatedDefault<T>(this List<T> list, int count)
        {
            return Repeated(list, count, default(T));
        }

        public static List<T> Repeated<T>(this List<T> list, int count, T value)
        {
            Guard.That(() => count).IsPositive();
            Guard.That(() => list).IsNotNull();            

            list.AddRange(Enumerable.Repeat(value, count));
            return list;
        }
    }
}
