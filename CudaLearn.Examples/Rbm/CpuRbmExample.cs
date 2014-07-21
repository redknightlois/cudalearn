using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra.Mkl;
using MathNet.Numerics.Random;
using MathNet.Numerics.Statistics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn.Examples.Rbm
{
    public static class CpuRbmExample
    {
        public static void Main(string[] args)
        {
            MathNet.Numerics.Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();

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

            var w_vh = 0.1f * Matrix<float>.Build.Random(num_vis, num_hid);
            var w_v = Vector<float>.Build.Dense(num_vis);
            var w_h = Vector<float>.Build.Dense(num_hid);

            // Initialize Weights updates
            var wu_vh = Matrix<float>.Build.Dense(num_vis, num_hid);
            var wu_v = Vector<float>.Build.Dense(num_vis);
            var wu_h = Vector<float>.Build.Dense(num_hid);

            var watch = new Stopwatch();

            var generator = new ContinuousUniform();

            var error = new double[num_epochs];
            for (int epoch = 0; epoch < num_epochs; epoch++)
            {
                Console.WriteLine(string.Format("Epoch {0}", epoch + 1));

                int batch = 0;
                foreach (var v_true in trainingSet.Batches(batch_size))
                {
                    Matrix<float> v = v_true.Item1;

                    // Apply momentum
                    wu_vh *= momentum;
                    wu_v *= momentum;
                    wu_h *= momentum;                    

                    // Positive phase                      
                    var h = 1.0f / (1 + ((-w_vh.Transpose() * v).AddVectorOnEachRow(w_h)).PointwiseExp());

                    wu_vh += v * h.Transpose();
                    wu_v += v.RowSums();
                    wu_h += h.RowSums(); 

                    // Sample hiddens
                    h = Functions.BinaryLess(h, Matrix<float>.Build.Dense(num_hid, batch_size, (i, j) => (float)generator.Sample()));

                    // Negative phase
                    v = 1.0f / (1 + ((-(w_vh * h).AddVectorOnEachRow(w_v))).PointwiseExp());
                    h = 1.0f / (1 + ((-(w_vh.Transpose() * v).AddVectorOnEachRow( w_h ))).PointwiseExp());

                    wu_vh -= v * h.Transpose();
                    wu_v -= v.RowSums(); 
                    wu_h -= h.RowSums(); 

                    // Update weights
                    w_vh += epsilon / batch_size * wu_vh;
                    w_v += epsilon / batch_size * wu_v;
                    w_h += epsilon / batch_size * wu_h;

                    error[epoch] = Distance.MSE(v.Storage.ToColumnMajorArray(), v_true.Item1.Storage.ToColumnMajorArray());
                    batch++;

                    Console.WriteLine(string.Format("Batch {0}/{1} | Error: {2}", batch, num_batches, error[epoch]));
                }
            }

            Console.WriteLine(string.Format("Mean squared error: {0}", ArrayStatistics.Mean(error)));
            Console.WriteLine(string.Format("Time: {0}", watch.Elapsed));
            Console.ReadLine();
        }
    }
}
