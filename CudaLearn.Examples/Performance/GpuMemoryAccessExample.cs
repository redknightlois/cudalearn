using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn.Examples.Performance
{
    public static class GpuMemoryAccessExample
    {
        private static Random generator = new Random();
        private static CudaDeviceVariable<float> Create(int size)
        {
            var aux = new CudaDeviceVariable<float>(size);
            for (int i = 0; i < aux.Size; i++)
                aux[i] = generator.Next(150) / 150.0f;

            return aux;
        }

        public static void Main(string[] args)
        {
            CudaLearnModule.Initialize();
            CudaLearnModule.AllowHandyForDebugButVerySlowGpuMemoryAccess = true;

            int rows = 1;
            int columns = 1000000;

            Console.WriteLine("Generating 1M random elements on the CPU");
            Console.WriteLine();

            var watch = new Stopwatch();
            watch.Restart();

            var m = new GpuMatrix<double>(rows, columns);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Columns; j++)
                    m[i, j] = generator.Next(150) / 150.0f;

            Console.WriteLine("GPU Naked Set per [i,j] = " + watch.ElapsedMilliseconds);
            watch.Restart();

            m = new GpuMatrix<double>(rows, columns);
            using (var s = new MemoryAccessScope<double>(m))
            {
                for (int i = 0; i < m.Rows; i++)
                    for (int j = 0; j < m.Columns; j++)
                        s[i, j] = generator.Next(150) / 150.0f;
            }

            Console.WriteLine("GPU MemoryScope Set per [i,j] = " + watch.ElapsedMilliseconds);
            watch.Restart();

            var mc = new Matrix<double>(rows, columns);
            for (int i = 0; i < mc.Rows; i++)
                for (int j = 0; j < mc.Columns; j++)
                    mc[i, j] = generator.Next(150) / 150.0f;

            Console.WriteLine("CPU Naked Set per [i,j] = " + watch.ElapsedMilliseconds);
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
        }
    }
}
