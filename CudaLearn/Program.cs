using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class Program
    {
        private static Random generator = new Random();
        private static CudaDeviceVariable<float> Create(int size)
        {
            var aux = new CudaDeviceVariable<float>(size);
            for (int i = 0; i < aux.Size; i++)
                aux[i] = generator.Next(150) / 150.0f;

            return aux;
        }

        static void Main(string[] args)
        {
            var context = new CudaContext();

            int numElements = 4096;

            var input1 = Create(numElements);
            var input2 = Create(numElements);
            var output = new CudaDeviceVariable<float>(numElements);

            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
            
            var kernel = context.LoadKernelPTX("vectorAdd.ptx", "vectorAdd");
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);
            Console.WriteLine( string.Format( "X:{0} Y:{1} Z:{2}", kernel.GridDimensions.x, kernel.GridDimensions.y, kernel.GridDimensions.z ));

            kernel.Run(input1.DevicePointer, input2.DevicePointer, output.DevicePointer, numElements);

            for (int i = 0; i < numElements; i++)
                Console.Write(output[i] + ",");
            Console.WriteLine();
        }
    }
}
