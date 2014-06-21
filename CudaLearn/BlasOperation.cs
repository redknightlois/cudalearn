using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public enum BlasOperation
    {
        NonTranspose = Operation.NonTranspose,
        Transpose = Operation.Transpose,
    }
}
