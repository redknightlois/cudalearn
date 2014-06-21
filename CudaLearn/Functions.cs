using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class Functions
    {
        public static T Mean<T>(this Matrix<T> m, Axis axis) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            throw new NotImplementedException();
        }

        public static T Mean<T>(this GpuMatrix<T> m, Axis axis) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            throw new NotImplementedException();
        }

        public static Matrix<T> Sum<T>(this Matrix<T> m, Axis axis) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            throw new NotImplementedException();
        }

        public static GpuMatrix<T> Sum<T>(this GpuMatrix<T> m, Axis axis) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            throw new NotImplementedException();
        }

        public static T Dot<T>(this Matrix<T> m1, Matrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);

            throw new NotImplementedException();
        }

        public static T Dot<T>(this GpuMatrix<T> m1, GpuMatrix<T> m2) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m1 != null);
            Contract.Requires<ArgumentNullException>(m2 != null);

            throw new NotImplementedException();
        }

        public static Matrix<T> Exp<T>(this Matrix<T> m) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            throw new NotImplementedException();
        }

        public static GpuMatrix<T> Exp<T>(this GpuMatrix<T> m) where T : struct
        {
            Contract.Requires<ArgumentNullException>(m != null);

            throw new NotImplementedException();
        }
    }
}
