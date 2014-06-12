using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public static class BlasMath 
    {
        public static double AbsoluteSum<T>(GpuMatrix<T> matrix) where T : struct
        {
            Contract.Requires<ArgumentNullException>(matrix != null);
            Contract.Requires(matrix.Rows == 1);

            var storage = (IGpuMatrixStorage<T>)matrix;
            var blas = CudaLearnModule.BlasContext;
            var data = storage.GetDeviceMemory();

            if (typeof(T) == typeof(float))
            {
                return blas.AbsoluteSum(data as CudaDeviceVariable<float>, matrix.Rows);
            }
            else if (typeof(T) == typeof(double))
            {
                return blas.AbsoluteSum(data as CudaDeviceVariable<double>, matrix.Rows);
            }

            throw new NotSupportedException("Type: {0} is not supported by the BLAS library.");
        }

        public static double AbsoluteSum<T>(Matrix<T> matrix) where T : struct
        {
            Contract.Requires<ArgumentNullException>(matrix != null);
            Contract.Requires(matrix.Rows == 1);

            throw new NotImplementedException();
        }

    }
}
