using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class MemoryAccessScope<T> : IDisposable where T : struct
    {
        private readonly GpuMatrix<T> matrix;        
        private readonly IGpuMatrixStorage<T> storage;
        
        private readonly Matrix<T> localMatrix;

        public MemoryAccessScope ( GpuMatrix<T> matrix )
        {
            this.matrix = matrix;
            this.storage = (IGpuMatrixStorage<T>)matrix;
            this.localMatrix = (Matrix<T>)this.matrix;

            this.storage.Lock();
        }


        public T this[int iRow, int iCol]      // Access this matrix as a 2D array
        {
            get
            {
                return localMatrix[iRow, iCol];
            }
            set
            {
                localMatrix[iRow, iCol] = value;
            }
        }

        public void Dispose()
        {
            var deviceMemory = this.storage.GetDeviceMemory();
            var hostMemory = ((IHostMatrixStorage<T>)this.localMatrix).GetHostMemory();
            deviceMemory.CopyToDevice(hostMemory);

            this.storage.Unlock();
        }
    }
}
