using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaDnn
{
    public enum CudnnTensorFormat
    {
        /// <summary>
        /// NCHW - row major (wStride = 1, hStride = w)
        /// This tensor format specifies that the data is laid out in the following order: image, features map, rows, columns. The strides are implicitly defined
        /// in such a way that the data are contiguous in memory with no padding between images, feature maps, rows, and columns; the columns are the
        /// inner dimension and the images are the outermost dimension.
        /// </summary>
        MajorRow = 0, 
        /// <summary>
        /// NHWC - feature maps interleaved ( cStride = 1 )
        /// This tensor format specifies that the data is laid out in the following order: image, rows, columns, features maps. The strides are implicitly defined in
        /// such a way that the data are contiguous in memory with no padding between images, rows, columns, and features maps; the feature maps are the
        /// inner dimension and the images are the outermost dimension.
        /// </summary>
        Interleaved = 1, 

        /// <summary>
        /// NCHW - row major (wStride = 1, hStride = w)
        /// This tensor format specifies that the data is laid out in the following order: image, features map, rows, columns. The strides are implicitly defined
        /// in such a way that the data are contiguous in memory with no padding between images, feature maps, rows, and columns; the columns are the
        /// inner dimension and the images are the outermost dimension.
        /// </summary>
        NCHW = 0,
        /// <summary>
        /// NHWC - feature maps interleaved ( cStride = 1 )
        /// This tensor format specifies that the data is laid out in the following order: image, rows, columns, features maps. The strides are implicitly defined in
        /// such a way that the data are contiguous in memory with no padding between images, rows, columns, and features maps; the feature maps are the
        /// inner dimension and the images are the outermost dimension.
        /// </summary>
        NHWC = 1,
    }
}
