using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn.Examples.Rbm
{
    public static class GpuRbmExample
    {
        public static void Main(string[] args)
        {
            CudaLearnModule.Initialize();

            var database = new MnistDatabase(new DirectoryInfo("."));
            var trainingSet = database.GetTrainingSet();

            // Training parameters
            float epsilon = 0.01f;
            float momentum = 0.9f;

            int num_epochs = 10;
            int batch_size = 100;

            int num_batches = trainingSet.Samples / batch_size;

            // Training parameters
            int num_vis = trainingSet.SampleSize;
            int num_hid = 1024;

            // Initialize Weights
            var w_vh = 0.1f * GpuMatrix<float>.Normal(num_vis, num_hid);
            var w_v = GpuMatrix<float>.Zeroes(num_vis, 1);
            var w_h = GpuMatrix<float>.Zeroes(num_hid, 1);

            // Initialize Weights updates
            var wu_vh = GpuMatrix<float>.Zeroes(num_vis, num_hid);
            var wu_v = GpuMatrix<float>.Zeroes(num_vis, 1);
            var wu_h = GpuMatrix<float>.Zeroes(num_hid, 1);

            var watch = new Stopwatch();

            var error = new float[num_epochs];
            for (int epoch = 0; epoch < num_epochs; epoch++)
            {
                Console.WriteLine(string.Format("Epoch {0}", epoch + 1));

                int batch = 0;
                foreach (var v_true in trainingSet.Batches(batch_size))
                {
                    GpuMatrix<float> v = v_true.Item1;

                    // Apply momentum
                    wu_vh *= momentum;
                    wu_v *= momentum;
                    wu_h *= momentum;

                    // Positive phase                    
                    var h = 1.0f / (1 + Functions.Exp(-(BlasMath.Gemm(1.0f, w_vh, v, BlasOperation.Transpose, BlasOperation.NonTranspose) + w_h)));

                    wu_vh += BlasMath.Gemm(1.0f, v, h, BlasOperation.NonTranspose, BlasOperation.Transpose);
                    wu_v += Functions.Sum(v, Axis.Rows);
                    wu_h += Functions.Sum(h, Axis.Rows);

                    // Sample hiddens
                    h = Functions.BinaryLess(h, GpuMatrix<float>.Uniform(num_hid, batch_size));

                    // Negative phase
                    v = 1.0f / (1 + Functions.Exp(-(w_vh * h + w_v)));
                    h = 1.0f / (1 + Functions.Exp(-(BlasMath.Gemm(1.0f, w_vh, v, BlasOperation.Transpose, BlasOperation.NonTranspose) + w_h)));

                    wu_vh -= BlasMath.Gemm(1.0f, v, h, BlasOperation.NonTranspose, BlasOperation.Transpose);
                    wu_v -= Functions.Sum(v, Axis.Rows);
                    wu_h -= Functions.Sum(h, Axis.Rows);

                    // Update weights
                    w_vh += epsilon / batch_size * wu_vh;
                    w_v += epsilon / batch_size * wu_v;
                    w_h += epsilon / batch_size * wu_h;

                    error[epoch] = Functions.Mean((v - v_true.Item1).ElementwisePow(2));
                    batch++;

                    Console.WriteLine(string.Format("Batch {0}/{1} | Error: {2}", batch, num_batches, error[epoch]));
                }
            }

            Console.WriteLine(string.Format("Mean squared error: {0}", Functions.Mean(error.AsMatrix())));
            Console.WriteLine(string.Format("Time: {0}", watch.Elapsed));
            Console.ReadLine();

            CudaLearnModule.Release();
        }
    }
}
