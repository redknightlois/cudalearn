using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CudaLearn.Tests
{
    public class GradientChecker
    {       
        private readonly float step;
        private readonly float threshold;
        private readonly int seed;
        private readonly float kink;
        private readonly float kinkRange;

        // kink and kink_range specify an ignored nonsmooth region of the form
        // kink - kink_range <= |feature value| <= kink + kink_range,
        // which accounts for all nonsmoothness in use
        public GradientChecker(float step, float threshold, int seed = 1701, float kink = 0.0f, float kinkRange = -1.0f)
        {
            this.step = step;
            this.threshold = threshold;
            this.seed = seed;
            this.kink = kink;
            this.kinkRange = kinkRange;
        }

        public void Check ( Layer layer, Blob bottom, Blob top, int checkBottom = -1)
        {            
            this.Check ( layer, new List<Blob> { bottom }, new List<Blob> { top }, checkBottom );
        }

        public void Check ( Layer layer, IList<Blob> bottom, IList<Blob> top, int checkBottom = -1)
        {
            layer.Setup( bottom, top );
            CheckSingle( layer, bottom, top, checkBottom, -1, -1);
        }


        public void CheckExhaustive ( Layer layer, Blob bottom, Blob top, int checkBottom = -1)
        {
            this.CheckExhaustive ( layer, new List<Blob> { bottom }, new List<Blob> { top }, checkBottom );
        }

        public void CheckExhaustive( Layer layer, IList<Blob> bottom, IList<Blob> top, int checkBottom = -1)
        {
            layer.Setup( bottom, top );
            Assert.True(top.Count > 0, "Exhaustive mode requires at least one top blob.");

            for (int i = 0; i < top.Count; i++)
                for (int j = 0; j < top[i].Count; j++)
                    CheckSingle(layer, bottom, top, checkBottom, i, j);
        }

        public void CheckEltwise ( Layer layer, Blob bottom, Blob top)
        {
            this.CheckEltwise ( layer, new List<Blob> { bottom }, new List<Blob> { top });
        }


        public void CheckEltwise ( Layer layer, IList<Blob> bottom, IList<Blob> top )
        {
            layer.Setup(bottom, top);
            Assert.True(top.Count > 0, "Exhaustive mode requires at least one top blob.");

            int checkBottom = -1;
            for (int i = 0; i < top.Count; i++)
                for (int j = 0; j < top[i].Count; j++)
                    CheckSingle(layer, bottom, top, checkBottom, i, j, elementwise: true);
        }
        public void CheckSingle( Layer layer,  Blob bottom, Blob top, int checkBottom, int topId, int topDataId, bool elementWise = false)
        {
            this.CheckSingle ( layer, new List<Blob> { bottom }, new List<Blob> { top }, checkBottom, topId, topDataId, elementWise );
        }

        public void CheckSingle( Layer layer,  IList<Blob> bottom, IList<Blob> top, int checkBottom, int topId, int topDataId, bool elementwise = false)
        {
            //TODO If implemented at all the ability of the layer to access stored blobs, we need to recheck this.
            if ( elementwise )
            {
                Assert.True(topId >= 0);
                Assert.True(topDataId >= 0);
                
                int topCount = top[topId].Count;
                for (int blobId = 0; blobId < bottom.Count; blobId++)
                    Assert.Equal(topCount, bottom[blobId].Count);
            }

            // First, figure out what blobs we need to check against.
            var blobsToCheck = new List<Blob>();
            var propagateDown = new List<bool>().Repeated(bottom.Count, checkBottom < 0);
            if ( checkBottom < 0 )
            {
                // We are not checking the bottom.
                for (int i = 0; i < bottom.Count; i++)
                    blobsToCheck.Add(bottom[i]);
            }
            else
            {
                // We are checking the bottom, therefore we must ensure that the blob checked exists.
                Assert.True(checkBottom < bottom.Count);
                blobsToCheck.Add(bottom[checkBottom]);
                propagateDown[checkBottom] = true;
            }

            //TODO Add a general random generator that layers should use, to ensure we always apply it when layers are non-deterministic.

            // Compute the gradient analytically using Backward
            // Get any loss from the layer
            float computedObjective = layer.Forward(bottom, top);

            // Get additional loss from the objective
            computedObjective += GetObjectiveAndGradient(top, topId, topDataId);
            layer.Backward(top, propagateDown, bottom);

            // Store computed gradients for all checked blobs
            var computedGradientsBlob = new Blob[blobsToCheck.Count];
            for ( int blobId = 0; blobId < blobsToCheck.Count; blobId++ )
            {
                var currentBlob = blobsToCheck[blobId];
                computedGradientsBlob[blobId] = new Blob(currentBlob);

                var currentDiff = currentBlob.Diff;
                var computedGradients = computedGradientsBlob[blobId].Data;                
                currentDiff.CopyTo(computedGradients);
            }

            // Compute derivative of top w.r.t. each bottom and parameter input using
            // finite differencing.

            for (int blobId = 0; blobId < blobsToCheck.Count; blobId++ )
            {
                var currentBlob = blobsToCheck[blobId];
                var computedGradients = computedGradientsBlob[blobId].Data;
                for ( int featId = 0; featId < currentBlob.Count; featId++ )
                {
                    // For an element-wise layer, we only need to do finite differencing to
                    // compute the derivative of topData[top_id][top_data_id] w.r.t.
                    // bottomData[blob_id][i] only for i == top_data_id.  For any other
                    // i != top_data_id, we know the derivative is 0 by definition, and simply
                    // check that that's true.
                    float estimatedGradient = 0;
                    if (!elementwise || featId == topDataId)
                    {
                        //TODO Add a general random generator that layers should use, to ensure we always apply it when layers are non-deterministic.

                        // Do finite differencing.
                        // Compute loss with stepsize added to input.
                        currentBlob.Data[featId] += step;
                        float positiveObjective = layer.Forward(bottom, top);
                        positiveObjective += GetObjectiveAndGradient(top, topId, topDataId);

                        // Compute loss with stepsize subtracted from input.
                        currentBlob.Data[featId] -= step * 2;

                        //TODO Add a general random generator that layers should use, to ensure we always apply it when layers are non-deterministic.

                        float negativeObjective = layer.Forward(bottom, top);
                        negativeObjective += GetObjectiveAndGradient(top, topId, topDataId);

                        // Recover original input value.
                        currentBlob.Data[featId] += step;
                        estimatedGradient = (positiveObjective - negativeObjective) / step / 2.0f;
                    }

                    float computedGradient = computedGradients[featId];
                    float feature = currentBlob.Data[featId];
                    if ( kink - kinkRange > Math.Abs(feature) || Math.Abs(feature) > kink + kinkRange )
                    {
                        // We check relative accuracy, but for too small values, we threshold
                        // the scale factor by 1

                        float scale = Math.Max( Math.Max(Math.Abs(computedGradient), Math.Abs(estimatedGradient)), 1.0f);
                        Assert.InRange(computedGradient - estimatedGradient, -threshold * scale, threshold * scale);
                    }
                }
            }

        }

        private float GetObjectiveAndGradient(IList<Blob> top, int topId, int topDataId)
        {
            float loss = 0;
            if ( topId < 0 )
            {
                // the loss will be half of the sum of squares of all outputs
                for (int i = 0; i < top.Count; i++)
                {
                    var topBlob = top[i];
                    int count = topBlob.Count;
                    for ( int j = 0; j < count; j++ )
                        loss += topBlob.Data[j] * topBlob.Data[j];

                    topBlob.Data.CopyTo(topBlob.Diff);
                }
                loss /= 2.0f;
            }
            else
            {
                // the loss will be the top_data_id-th element in the top_id-th blob.
                for (int i = 0; i < top.Count; i++)
                    top[i].Diff.Clear();

                loss = top[topId].Data[topDataId];
                top[topId].Diff[topDataId] = 1.0f;
            }
            return loss;
        }
    }
}
