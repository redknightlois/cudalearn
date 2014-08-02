using MathNet.Numerics.LinearAlgebra;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn
{
    public class Blob
    {
        private Vector<float> _data;
        private Vector<float> _diff;

        public Vector<float> Data 
        {
            get
            {
                if (this.Count == 0)
                    throw new InvalidOperationException("Blob not initialized yet. You need to reshape it with a non zero size.");

                return _data;
            }
            private set
            {
                _data = value;
            }
        }

        public Vector<float> Diff 
        { 
            get
            {
                if (this.Count == 0)
                    throw new InvalidOperationException("Blob not initialized yet. You need to reshape it with a non zero size.");

                return _diff;
            }
            private set
            {
                this._diff = value;
            }
        }

        public Blob() : this(0, 0, 0, 0)
        { }

        public Blob( Blob blob )
        {
            Guard.That(() => blob).IsNotNull();

            ReshapeAs(blob);
        }

        public Blob(int num, int channels, int height, int width)
        {
            Reshape(num, channels, height, width);
        }

        public void Reshape(int num, int channels, int height, int width)
        {
            Guard.That(() => num).IsNatural();
            Guard.That(() => channels).IsNatural();
            Guard.That(() => height).IsNatural();
            Guard.That(() => width).IsNatural();

            this.Num = num;
            this.Channels = channels;
            this.Height = height;
            this.Width = width;

            if ( this.Count != 0 )
            {
                this.Data = Vector<float>.Build.Dense(this.Count);
                this.Diff = Vector<float>.Build.Dense(this.Count);
            }
            else
            {
                this.Data = null;
                this.Diff = null;
            }
        }

        public void ReshapeAs( Blob other )
        {
            this.Reshape(other.Num, other.Channels, other.Height, other.Width);
        }

        public int Num { get; private set; }
        public int Channels { get; private set; }
        public int Height { get; private set; }
        public int Width { get; private set; }

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Num * this.Channels * this.Height * this.Width; }
        }        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Offset ( int n, int c = 0, int h = 0, int w = 0 )
        {
            ValidateAccessParameters(n, c, h, w);
            return ((n * this.Channels + c) * this.Height + h) * this.Width + w;    
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DataAt(int i)
        {
            ValidateAccessParameters(i);
            return this.Data.At(i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DiffAt(int i)
        {
            ValidateAccessParameters(i);
            return this.Diff.At(i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DataAt(int n, int c, int h, int w)
        {
            ValidateAccessParameters(n, c, h, w);
            return this.Data.At(Offset(n, c, h, w));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DiffAt(int n, int c, int h, int w )
        {
            ValidateAccessParameters(n, c, h, w);
            return this.Diff.At(Offset(n, c, h, w));
        }

#region Validation methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateAccessParameters(int n, int c, int h, int w)
        {
#if EXHAUSTIVE_DEBUG
            Guard.That(() => n).IsInRange(0, this.Num);
            Guard.That(() => c).IsLessOrEqualThan(this.Channels);
            Guard.That(() => h).IsLessOrEqualThan(this.Height);
            Guard.That(() => w).IsLessOrEqualThan(this.Width);
#endif
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateAccessParameters(int i)
        {
#if EXHAUSTIVE_DEBUG
            Guard.That(() => i).IsInRange(0, this.Count);
#endif
        }

#endregion

        public void CopyFrom ( Blob other, bool copyDiff = false, bool reshape = false)
        {
            // If reshaping needed we reshape the instance with new memory.
            if ( reshape )
                ReshapeAs(other);

            // We copy the data
            other.Data.CopyTo( this.Data );

            // If copying differential is needed, we copy it too.
            if (copyDiff)
                other.Diff.CopyTo(this.Diff);
        }
    }
}
