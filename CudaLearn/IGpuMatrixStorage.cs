using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    /// <summary>
    /// This interface is only intended for advanced uses only. 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IGpuMatrixStorage<T> : IFluentInterface where T : struct
    {
        CudaDeviceVariable<T> GetDeviceMemory();

        void Lock();

        void Unlock();
    }
}
