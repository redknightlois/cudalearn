using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics.Contracts;

namespace CudaLearn
{
    public class Decomposition<T> where T : struct
    {
        public Matrix<T> L;
        public Matrix<T> U;
        public int[] Permutation;
        public T DeterminantOfP;

        public Decomposition( Matrix<T> l, Matrix<T> u, int[] permutation, T detOfP )
        {
            Contract.Requires(l != null);
            Contract.Requires(u != null);
            Contract.Requires(permutation != null);
            Contract.Requires(l.Rows == permutation.Length);
            Contract.Requires(l.Rows == u.Rows && l.Columns == u.Columns);

            this.L = l;
            this.U = u;
            this.Permutation = permutation;
            this.DeterminantOfP = detOfP;
        }
    }
}
