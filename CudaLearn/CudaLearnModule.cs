using ManagedCuda;
using ManagedCuda.CudaBlas;
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

        public static CudaContext Context
        {
            get { return _context; }
        }

        public static CudaBlas BlasContext
        {
            get { return _blasContext; }
        }

        public static bool AllowHandyForDebugButVerySlowGpuMemoryAccess = false;

        public static void Initialize ()
        {
            _context = new CudaContext();
            _blasContext = new CudaBlas();
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
        }


    }
}
