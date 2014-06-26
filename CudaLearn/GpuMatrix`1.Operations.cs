﻿using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    partial class GpuMatrix<T> : IEquatable<GpuMatrix<T>>, IGpuMatrixStorage<T>, IDisposable where T : struct
    {

        /// <summary>
        /// Creates a zero matrix
        /// </summary>
        /// <param name="iRows"></param>
        /// <param name="iCols"></param>
        /// <returns></returns>
        public static GpuMatrix<T> Zeroes(int iRows, int iCols)
        {
            Contract.Requires<ArgumentException>(iRows > 0 && iCols > 0);

            return new GpuMatrix<T>(iRows, iCols, GpuMatrix<T>.Zero);
        }

        /// <summary>
        /// Creates an identity matrix.
        /// </summary>
        public static GpuMatrix<T> Identity(int iRows)
        {
            Contract.Requires<ArgumentException>(iRows > 0);

            var m = new GpuMatrix<T>(iRows, iRows);
            BlasMath.SetIdentity(ref m);

            return m;
        }

        public static GpuMatrix<T> Uniform(int sizeX)
        {
            return Uniform(sizeX, GpuMatrix<T>.Zero, GpuMatrix<T>.One);
        }
        public static GpuMatrix<T> Uniform(int sizeX, T min, T max)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0);
            throw new NotImplementedException();
        }

        public static GpuMatrix<T> Uniform(int sizeX, int sizeY)
        {
            return Uniform(sizeX, sizeY, GpuMatrix<T>.Zero, GpuMatrix<T>.One);
        }

        public static GpuMatrix<T> Uniform(int sizeX, int sizeY, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0 && sizeY > 0);
            throw new NotImplementedException();
        }

        public static GpuMatrix<T> Normal(int sizeX)
        {
            return Normal(sizeX, GpuMatrix<T>.Zero, GpuMatrix<T>.One);
        }

        public static GpuMatrix<T> Normal(int sizeX, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0);
            throw new NotImplementedException();
        }

        public static GpuMatrix<T> Normal(int sizeX, int sizeY)
        {
            return Normal(sizeX, sizeY, GpuMatrix<T>.Zero, GpuMatrix<T>.One);
        }

        public static GpuMatrix<T> Normal(int sizeX, int sizeY, T mean, T deviation)
        {
            Contract.Requires<ArgumentOutOfRangeException>(sizeX > 0 && sizeY > 0);
            throw new NotImplementedException();
        }

        private static GpuMatrix<T> AddColumns(GpuMatrix<T> m1, GpuMatrix<T> v)
        {
            throw new NotImplementedException();
        }

        private static GpuMatrix<T> AddRows(GpuMatrix<T> m1, GpuMatrix<T> v)
        {
            throw new NotImplementedException();
        }
    }
}
