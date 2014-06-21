using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public enum Axis
    {
        None = 0, // Will treat a Matrix<T> as a vector.
        Columns = 1, // Will apply the function over the data in the column.
        Rows = 2, // Will apply the function over the data in the row.       
    }
}
