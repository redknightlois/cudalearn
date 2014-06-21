using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    partial class Matrix<T> : IEquatable<Matrix<T>>, IHostMatrixStorage<T>, IDisposable
        where T : struct
    {
        public static Matrix<T> Uniform(int sizeX)
        {
            return Uniform(sizeX, Matrix<T>.Zero, Matrix<T>.One);
        }
        public static Matrix<T> Uniform(int sizeX, T min, T max)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0);
            throw new NotImplementedException();
        }

        public static Matrix<T> Uniform(int sizeX, int sizeY)
        {
            return Uniform(sizeX, sizeY, Matrix<T>.Zero, Matrix<T>.One);
        }

        public static Matrix<T> Uniform(int sizeX, int sizeY, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0 && sizeY > 0);
            throw new NotImplementedException();
        }

        public static Matrix<T> Normal(int sizeX)
        {
            return Normal(sizeX, Matrix<T>.Zero, Matrix<T>.One);
        }

        public static Matrix<T> Normal(int sizeX, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0);
            throw new NotImplementedException();
        }

        public static Matrix<T> Normal(int sizeX, int sizeY)
        {
            return Normal(sizeX, sizeY, Matrix<T>.Zero, Matrix<T>.One);
        }

        public static Matrix<T> Normal(int sizeX, int sizeY, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0 && sizeY > 0);
            throw new NotImplementedException();
        }

    }
}
