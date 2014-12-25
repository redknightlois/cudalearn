using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Providers.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class InnerProductLayerConfiguration : LayerConfiguration
    {
        public InnerProductLayerConfiguration(int outputs, bool bias = true, FillerConfiguration weightsFiller = null, FillerConfiguration biasFiller = null)
            : base(LayerType.Dropout)
        {
            Contract.Requires(outputs > 0);

            Guard.That(() => outputs).IsGreaterThan(0);

            this.Outputs = outputs;
            this.UseBias = bias;

            this.WeightsFiller = weightsFiller ?? new XavierFillerConfiguration();
            this.BiasFiller = biasFiller ?? new ConstantFillerConfiguration();
        }

        public int Outputs { get; set; }
        public bool UseBias { get; set; }

        public FillerConfiguration WeightsFiller { get; set; }

        public FillerConfiguration BiasFiller { get; set; }
    }

    /// <summary>
    /// InnerProductLayer
    ///        Fully connected layer.
    /// </summary>
    public class InnerProductLayer : NeuronLayer<InnerProductLayerConfiguration>
    {
        private Tensor weights;        // n x k
        private Tensor bias;           // 1 x n
        private Tensor biasMultiplier; // 1 x m

        // bottom: m x k
        // top: m x n

        private int n;
        private int m;
        private int k;


        public InnerProductLayer(int outputs, bool bias = true, FillerConfiguration weightsFiller = null, FillerConfiguration biasFiller = null)
            : this(new InnerProductLayerConfiguration(outputs, bias, weightsFiller, biasFiller))
        { }

        public InnerProductLayer(InnerProductLayerConfiguration param)
            : base(param)
        { }

        public override void Setup(TensorCollection bottom, TensorCollection top)
        {
            base.Setup(bottom, top);

            this.n = Parameters.Outputs;
            this.m = bottom[0].Num;
            this.k = bottom[0].Count / bottom[0].Num;

            // Reshape the output
            top[0].Reshape(m, n, 1, 1);

            if (this.weights == null || this.bias == null)
            {
                // Fill the weights
                this.weights = new Tensor(1, 1, n, k);
                var weightsFiller = FillerFactory.Create(Parameters.WeightsFiller);
                weightsFiller.Fill(weights);

                // If necessary, initialize and fill the bias term
                if (Parameters.UseBias)
                {
                    this.bias = new Tensor(1, 1, 1, n);
                    var biasFiller = FillerFactory.Create(Parameters.BiasFiller);
                    biasFiller.Fill(bias);
                }
            }
            else
            {
                // LOG we are skipping the parameter initialization
            }

            if (Parameters.UseBias)
            {
                this.biasMultiplier = new Tensor(1, 1, 1, m);
                using (var biasMultiplierCpu = this.biasMultiplier.OnCpu())
                {
                    biasMultiplierCpu.Data.Map(v => 1, biasMultiplierCpu.Data, Zeros.Include);
                }
            }
        }

        internal override double ForwardCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
        {
            var provider = Control.LinearAlgebraProvider;

            using (var weightsCpu = this.weights.OnCpu())          
            {
                var bottomData = (DenseVector)bottom[0].Data;
                var topData = (DenseVector)top[0].Data;
                var weightsData = (DenseVector)weightsCpu.Data;

                provider.MatrixMultiplyWithUpdate(Transpose.DontTranspose, Transpose.Transpose, 1f, bottomData.Values, m, k, weightsData.Values, n, k, 0f, topData.Values);

                if (Parameters.UseBias)
                {
                    using (var biasCpu = this.bias.OnCpu())
                    using (var biasMultiplierCpu = this.biasMultiplier.OnCpu())
                    {
                        var biasData = (DenseVector)biasCpu.Data;
                        var biasMultiplierData = (DenseVector)biasMultiplierCpu.Data;
                        provider.MatrixMultiplyWithUpdate(Transpose.DontTranspose, Transpose.DontTranspose, 1f, biasMultiplierData.Values, m, 1, biasData.Values, 1, n, 1f, topData.Values);
                    }
                }
            }
 
            return 0;
        }

        internal override void BackwardCpu(CpuTensorScopeCollection top, IList<bool> propagateDown, CpuTensorScopeCollection bottom)
        {
            var provider = Control.LinearAlgebraProvider;

            using (var weightsCpu = this.weights.OnCpu())
            {
                var bottomData = (DenseVector)bottom[0].Data;
                var topDiff = (DenseVector)top[0].Diff;
                var weightsData = (DenseVector)weightsCpu.Data;

                if (GetPropagateDownForParameter(0))
                {
                    provider.MatrixMultiplyWithUpdate(Transpose.Transpose, Transpose.DontTranspose, 1f, topDiff.Values, m, n, bottomData.Values, m, k, 0f, weightsData.Values);
                }

                if (this.Parameters.UseBias && GetPropagateDownForParameter(1))
                {
                    using (var biasCpu = this.bias.OnCpu())
                    using (var biasMultiplierCpu = this.biasMultiplier.OnCpu())
                    {
                        var biasData = (DenseVector)biasCpu.Data;
                        var biasMultiplierData = (DenseVector)biasMultiplierCpu.Data;

                        provider.MatrixMultiplyWithUpdate(Transpose.Transpose, Transpose.DontTranspose, 1f, topDiff.Values, m, n, biasMultiplierData.Values, 1, m, 0f, biasData.Values);
                    }
                }

                if (propagateDown[0])
                {
                    provider.MatrixMultiplyWithUpdate(Transpose.DontTranspose, Transpose.DontTranspose, 1f, topDiff.Values, m, n, weightsData.Values, n, k, 0f, bottomData.Values);
                } 
            }           
        }
    }
}
