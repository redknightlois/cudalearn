using ManagedCuda;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaRand;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class CudaLearnModule
    {
        private static CudaContext _context;
        private static CudaBlas _blasContext;
        private static CudaRandDevice _randContext;

        public static CudaContext Context
        {
            get { return _context; }
        }

        public static CudaBlas BlasContext
        {
            get { return _blasContext; }
        }

        public static CudaRandDevice RandomContext
        {
            get { return _randContext; }
        }

        public static bool AllowHandyForDebugButVerySlowGpuMemoryAccess = false;

        public static void Initialize ()
        {
            _context = new CudaContext();
            _blasContext = new CudaBlas();
            _randContext = new CudaRandDevice(GeneratorType.PseudoDefault);
        }

        public static void Release()
        {
            if (_blasContext != null)
            {
                _blasContext.Dispose();
                _blasContext = null;
            }    

            if (_context != null)
            {
                _context.Dispose();
                _context = null;
            }

            if (_randContext != null )
            {
                _randContext.Dispose();
                _randContext = null;
            }
        }


    }
}
