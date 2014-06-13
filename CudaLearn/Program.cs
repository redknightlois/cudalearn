using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            CudaLearnModule.Initialize();
            CudaLearnModule.AllowHandyForDebugButVerySlowGpuMemoryAccess = true;

            int rows = 1;
            int columns = 100000;

            var watch = new Stopwatch();
            watch.Restart();

            var m = new GpuMatrix<double>(rows, columns);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Columns; j++)
                    m[i, j] = generator.Next(150) / 150.0f;

            Console.WriteLine("GPU Set per [i,j] = " + watch.ElapsedMilliseconds);
            watch.Restart();

            var mc = new Matrix<double>(rows, columns);
            for (int i = 0; i < mc.Rows; i++)
                for (int j = 0; j < mc.Columns; j++)
                    mc[i, j] = generator.Next(150) / 150.0f;

            Console.WriteLine("CPU Set per [i,j] = " + watch.ElapsedMilliseconds);
            watch.Restart();

            double sum = 0;
            for (int k = 0; k < 1000; k++)
            {
                sum = BlasMath.SumMagnitudes(m);
            }

            Console.WriteLine("GPU Blas sequential = " + watch.ElapsedMilliseconds);
            watch.Restart();

            sum = 0;
            for (int k = 0; k < 1000; k++)
            {
                for (int i = 0; i < mc.Rows; i++)
                {
                    for (int j = 0; j < mc.Columns; j++)
                    {
                        double aux = mc[i, j];
                        sum += aux >= 0 ? aux : -aux;
                    }
                }
            }

            Console.WriteLine("CPU Blas sequential = " + watch.ElapsedMilliseconds);
            watch.Restart();

            Console.WriteLine("Total Sum: " + sum);
            Console.ReadLine();


            var context = CudaLearnModule.Context;

            int numElements = 4096;

            var input1 = Create(numElements);
            var input2 = Create(numElements);
            var output = new CudaDeviceVariable<float>(numElements);

            int threadsPerBlock = context.GetDeviceInfo().MaxThreadsPerBlock;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

            var kernel = context.LoadKernelPTX("vectorOperations.ptx", "vectorAdd1f");
            kernel.BlockDimensions = new dim3(threadsPerBlock);
            kernel.GridDimensions = new dim3(blocksPerGrid);
            Console.WriteLine(string.Format("X:{0} Y:{1} Z:{2}", kernel.GridDimensions.x, kernel.GridDimensions.y, kernel.GridDimensions.z));

            kernel.Run(input1.DevicePointer, input2.DevicePointer, output.DevicePointer, numElements);

            for (int i = 0; i < numElements; i++)
                Console.Write(output[i] + ",");
            Console.WriteLine();
        }
    }
}
