﻿using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class EuclideanLossLayerConfiguration : LayerConfiguration
    {
        public EuclideanLossLayerConfiguration()
            : base(LayerType.EuclideanLoss)
        {}
    }

    /// <summary>
    ///  EuclideanLossLayer
    ///         Compute the L_2 distance between the two inputs.
    ///    
    ///  loss = (1/2 \sum_i (a_i - b_i)^2)
    ///  a' = 1/I (a - b)
    /// </summary>
    public class EuclideanLossLayer : LossLayer<EuclideanLossLayerConfiguration>
    {
        private Vector<double> difference;

        public EuclideanLossLayer()
            : this(new EuclideanLossLayerConfiguration())
        { }

        public EuclideanLossLayer(EuclideanLossLayerConfiguration param)
            : base(param)
        { }

        public override void Setup(TensorCollection bottom, TensorCollection top)
        {
            base.Setup(bottom, top);

            if (!top.Any())
                top.Add(new Tensor(bottom[0]));
        }

        protected override void CheckBlobCount(TensorCollection bottom, TensorCollection top)
        {
            base.CheckBlobCount(bottom, top);

            Guard.That(() => bottom).IsTrue(x => x[0].Channels == bottom[1].Channels, "Channels in both bottom blobs must be equal.");
            Guard.That(() => bottom).IsTrue(x => x[0].Height == bottom[1].Height, "Height in both bottom blobs must be equal.");
            Guard.That(() => bottom).IsTrue(x => x[0].Width == bottom[1].Width, "Width in both bottom blobs must be equal.");
        }

        internal override double ForwardCpu(CpuTensorScopeCollection bottom, CpuTensorScopeCollection top)
        {            
            difference = bottom[0].Data - bottom[1].Data;
            double loss = (difference.L2Norm() / (bottom[0].Count / 2));

            // If we are expecting a value we just set it up.
            if ( top.Count == 1 )
                top[0].Data[0] = loss;

            return loss;
        }

        internal override void BackwardCpu(CpuTensorScopeCollection top, IList<bool> propagateDown, CpuTensorScopeCollection bottom)
        {
            for (int i = 0; i < 2; i++)
            {
                if ( propagateDown[i] )
                {
                    double sign = (i == 0) ? 1 : -1;
                    double alpha = sign / bottom[i].Num;

                    var bottomDiff = bottom[i].Diff;
                    difference.Map(v => alpha * v, bottomDiff, Zeros.Include);
                }
            }
        }


        // Unlike most loss layers, in the EuclideanLossLayer we can back-propagate
        // to both inputs.
        protected override bool AllowForceBackward(int bottomIndex)
        {
            return true;
        }
    }
}
