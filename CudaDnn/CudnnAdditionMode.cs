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
        /// In this mode, the bias tensor is defined as one image with one feature map. This image will be added to every feature map of every image of the input/output tensor
        /// </summary>
        Image = 0,
        /// <summary>
        /// Add one image to every feature maps of each input
        /// In this mode, the bias tensor is defined as one image with one feature map. This image will be added to every feature map of every image of the input/output tensor 
        /// </summary>
        SameHW = 0,

        /// <summary>
        ///  Add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest
        ///  In this mode, the bias tensor is defined as one image with multiple feature maps. This image will be added to every image of the input/output tensor.
        /// </summary>
        FeatureMap = 1,
        /// <summary>
        ///  Add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest
        ///  In this mode, the bias tensor is defined as one image with multiple feature maps. This image will be added to every image of the input/output tensor.
        /// </summary>
        SameCHW = 1,
        
        /// <summary>
        /// Add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input 
        /// In this mode, the bias tensor is defined as one image with multiple feature maps of dimension 1x1; it can be seen as an vector of feature maps.
        /// Each feature map of the bias tensor will be added to the corresponding feature map of all height-by-width pixels of every image of the input/output tensor.
        /// </summary>
        SameC = 2,

        /// <summary>
        /// Add 2 tensors with same n,c,h,w
        /// In this mode, the bias tensor has the same dimensions as the input/output tensor. It will be added point-wise to the input/output tensor
        /// </summary>
        FullTensor = 3,
    }
}
