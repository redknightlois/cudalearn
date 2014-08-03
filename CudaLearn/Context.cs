using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaLearn
{
    public enum PhaseType
    {
        Train,
        Test
    }

    public enum ExecutionModeType
    {
        Automatic,
        Cpu,
        Gpu,
    }

    public class Context
    {
        private static Lazy<Context> _instance = new Lazy<Context>(SetupDefaultContext, true);

        public static Context Instance
        {
            get { return _instance.Value; }
        }

        private static Context SetupDefaultContext()
        {
            return new Context();
        }

        protected Context ()
        {
            this.Phase = PhaseType.Test;
            this.Mode = ExecutionModeType.Automatic;
        }

        public ExecutionModeType Mode { get; set; }

        public PhaseType Phase { get; set; }
    }
}
