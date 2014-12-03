using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;

namespace CudaDnn
{
    [Serializable]
    public class CudnnException : Exception, ISerializable  
    {
        public readonly CudnnStatus ErrorCode;

        public CudnnException(CudnnStatus error)
        {
            ErrorCode = error;
        }

        public CudnnException(CudnnStatus error, string message) 
            : base(message)
        {
            ErrorCode = error;
        }

        public CudnnException(CudnnStatus error, string message, Exception innerException)
            : base (message, innerException)
        {
            ErrorCode = error;
        }

        protected CudnnException(SerializationInfo info, StreamingContext context)
            : base (info, context)
        {}


        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue("ErrorCode", this.ErrorCode);
        }
    }
}
