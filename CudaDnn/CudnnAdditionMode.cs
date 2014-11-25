using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    public enum CudnnAdditionMode
    {
        /// <summary>
        /// Add one image to every feature maps of each input
        /// </summary>
        Image = 0,
        /// <summary>
        /// add one image to every feature maps of each input
        /// </summary>
        SameHW = 0,

        /// <summary>
        ///  add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest
        /// </summary>
        FeatureMap = 1,
        /// <summary>
        ///  add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest
        /// </summary>
        SameCHW = 1,
        
        /// <summary>
        /// add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input 
        /// </summary>
        SameC = 2,

        /// <summary>
        /// add 2 tensors with same n,c,h,w
        /// </summary>
        FullTensor = 3,
    }
}
