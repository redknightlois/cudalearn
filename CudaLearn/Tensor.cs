using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using Seterlund.CodeGuard;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace CudaLearn
{
    public enum TensorLocation
    {
        Cpu,
        Gpu
    }

    public class CpuTensorScope : IDisposable
    {
        internal readonly Tensor Tensor;

        public int Num
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Tensor.Num; }
        }
        public int Channels
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Tensor.Channels; }
        }

        public int Height
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Tensor.Height; }
        }
        public int Width
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Tensor.Width; }
        }

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return Tensor.Num * Tensor.Channels * Tensor.Height * Tensor.Width; }
        }        

        internal CpuTensorScope(Tensor @this, Vector<double> data, Vector<double> diff)
        {
            Contract.Requires(@this != null);
            Contract.Requires(data != null);
            Contract.Requires(diff != null);
            Contract.Requires(@this.Count != 0);

            this.Tensor = @this;
            this._data = data;
            this._diff = diff;
        }

        ~CpuTensorScope()
        {
            // Finalizer calls Dispose(false)
            Dispose(false);
        }

        // Dispose() calls Dispose(true)
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // The bulk of the clean-up code is implemented in Dispose(bool)
        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                // free managed resources
                Tensor.Unlock(TensorLocation.Cpu);
            }
        }


        private readonly Vector<double> _data;

        public Vector<double> Data
        {
            get { return _data; }
        }

        private readonly Vector<double> _diff;
        public Vector<double> Diff
        {
            get { return _diff; }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Offset(int n, int c = 0, int h = 0, int w = 0)
        {
            ValidateAccessParameters(n, c, h, w);
            return ((n * Tensor.Channels + c) * Tensor.Height + h) * Tensor.Width + w;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double DataAt(int i)
        {
            ValidateAccessParameters(i);
            return this.Data.At(i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double DiffAt(int i)
        {
            ValidateAccessParameters(i);
            return this.Diff.At(i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double DataAt(int n, int c, int h, int w)
        {
            ValidateAccessParameters(n, c, h, w);
            return this.Data.At(Offset(n, c, h, w));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double DiffAt(int n, int c, int h, int w)
        {
            ValidateAccessParameters(n, c, h, w);
            return this.Diff.At(Offset(n, c, h, w));
        }

        #region Validation methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateAccessParameters(int n, int c, int h, int w)
        {
#if EXHAUSTIVE_DEBUG
            Guard.That(() => n).IsInRange(0, Tensor.Num);
            Guard.That(() => c).IsLessOrEqualThan(Tensor.Channels);
            Guard.That(() => h).IsLessOrEqualThan(Tensor.Height);
            Guard.That(() => w).IsLessOrEqualThan(Tensor.Width);
#endif
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateAccessParameters(int i)
        {
#if EXHAUSTIVE_DEBUG
            Guard.That(() => i).IsInRange(0, Tensor.Count);
#endif
        }

        public void InitializeWith(double dataValue, double diffValue)
        {
            this.Data.Map(x => dataValue, this.Data, Zeros.Include);
            this.Diff.Map(x => diffValue, this.Diff, Zeros.Include);
        }
    }


    public class GpuTensorScope : IDisposable
    {
        internal readonly Tensor Tensor;

        public int Num
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Tensor.Num; }
        }
        public int Channels
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Tensor.Channels; }
        }

        public int Height
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Tensor.Height; }
        }
        public int Width
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this.Tensor.Width; }
        }

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return Tensor.Num * Tensor.Channels * Tensor.Height * Tensor.Width; }
        }    

        internal GpuTensorScope(Tensor @this)
        {
            Contract.Requires(@this != null);
            Contract.Requires(@this.Count != 0);

            this.Tensor = @this;
        }

        ~GpuTensorScope()
        {
            // Finalizer calls Dispose(false)
            Dispose(false);
        }

        // Dispose() calls Dispose(true)
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // The bulk of the clean-up code is implemented in Dispose(bool)
        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                // free managed resources
                Tensor.Unlock(TensorLocation.Gpu);
            }
        }
    }

    public class Tensor
    {
        public TensorLocation Location { get; private set; }


        private DenseVectorStorage<double> _data;
        private DenseVectorStorage<double> _diff;


        public Tensor() : this(0, 0, 0, 0)
        { }

        public Tensor( Tensor blob )
        {
            Guard.That(() => blob).IsNotNull();

            ReshapeAs(blob);
        }

        public Tensor(int num, int channels, int height, int width)
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

            if (this.Count != 0)
            {
                switch (this.Location)
                {
                    case TensorLocation.Cpu: // TODO We can do better here and do not reshape if the size is the same.                         
                        this._data = DenseVectorStorage<double>.OfValue(this.Count, 0.0f);
                        this._diff = DenseVectorStorage<double>.OfValue(this.Count, 0.0f);
                        break;
                    case TensorLocation.Gpu:
                        throw new NotImplementedException();
                }
            }
            else
            {
                this._data = null;
                this._diff = null;
            }
        }
        public void ReshapeAs( Tensor other )
        {
            if (other == null)
                throw new ArgumentNullException("other");
            Contract.EndContractBlock();

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

#endregion

        public void CopyFrom(Tensor other, bool copyDiff = false, bool reshape = false)
        {
            if (other == null)
                throw new ArgumentNullException("other");
            Contract.EndContractBlock();           

            // If reshaping needed we reshape the instance with new memory.
            if (reshape)
                ReshapeAs(other);

            switch (this.Location)
            {
                case TensorLocation.Cpu:
                    using (var @thisCpu = this.OnCpu())
                    using (var @otherCpu = other.OnCpu())
                    {
                        // We copy the data
                        @otherCpu.Data.CopyTo(@thisCpu.Data);

                        // If copying differential is needed, we copy it too.
                        if (copyDiff)
                            @otherCpu.Diff.CopyTo(@thisCpu.Diff);
                    }
                    break;
                case TensorLocation.Gpu:
                    break;
            }

            throw new NotImplementedException();


        }

        public void ShareData(Tensor other)
        {
            switch (this.Location)
            {
                case TensorLocation.Cpu:
                    this._data = other._data;
                    break;
                case TensorLocation.Gpu:
                    throw new NotImplementedException();
            }
        }

        public void ShareDiff(Tensor other)
        {
            switch (this.Location)
            {
                case TensorLocation.Cpu:
                    this._diff = other._diff;
                    break;
                case TensorLocation.Gpu:
                    throw new NotImplementedException();
            }
        }

        public void InitializeWith(double dataValue, double diffValue)
        {
            switch (this.Location)
            {
                case TensorLocation.Cpu:
                    using (var @thisCpu = this.OnCpu())
                    {
                        @thisCpu.Data.Map(x => dataValue, @thisCpu.Data, Zeros.Include);
                        @thisCpu.Diff.Map(x => diffValue, @thisCpu.Diff, Zeros.Include);
                    }

                    break;
                case TensorLocation.Gpu:
                    throw new NotImplementedException();
            }            
        }

        private volatile TensorLocation activeScopeLocation = TensorLocation.Cpu;
        private int activeScopeCount = 0;
        private readonly object syncRoot = new object();

        internal void Lock(TensorLocation location)
        {
            lock (syncRoot)
            {
                if (activeScopeLocation != location && activeScopeCount != 0)
                    throw new InvalidOperationException("Cannot request a tensor lock. There is already an scope active with a different location.");

                Interlocked.Increment(ref activeScopeCount);
                activeScopeLocation = location;
            }
        }

        internal void Unlock(TensorLocation location)
        {
            lock (syncRoot)
            {
                if (activeScopeLocation != location)
                    throw new InvalidOperationException("Cannot request a tensor unlock on a different location.");

                if (activeScopeCount == 0)
                    throw new InvalidOperationException("Cannot request a tensor unlock. There is no scope active to support the request.");

                Interlocked.Decrement(ref activeScopeCount);
                activeScopeLocation = location;
            }
        }

        public CpuTensorScope OnCpu()
        {
            if (this.Count == 0)
                throw new InvalidOperationException("Tensor is not initialized.");

            Contract.Ensures(Contract.Result<CpuTensorScope>() != null);
            Contract.EndContractBlock();

            lock (syncRoot)
            {
                try
                {
                    Lock(TensorLocation.Cpu);
                    switch (this.Location)
                    {
                        case TensorLocation.Cpu: return new CpuTensorScope(this, Vector<double>.Build.OfStorage(_data), Vector<double>.Build.OfStorage(_diff));
                        case TensorLocation.Gpu: throw new NotImplementedException();
                        default: throw new NotSupportedException("The location is not supported. Make sure you didn't add a new TensorLocation and haven't updated this code appropriately");
                    }
                }
                finally
                {
                    this.Location = TensorLocation.Cpu;
                }
            }
        }

        public GpuTensorScope OnGpu()
        {            
            if (this.Count == 0)
                throw new InvalidOperationException("Tensor is not initialized.");

            Contract.Ensures(Contract.Result<GpuTensorScope>() != null);
            Contract.EndContractBlock();

            lock (syncRoot)
            {
                try
                {
                    Lock(TensorLocation.Gpu);
                    switch (this.Location)
                    {
                        case TensorLocation.Cpu: throw new NotImplementedException();
                        case TensorLocation.Gpu: throw new NotImplementedException();
                        default: throw new NotSupportedException("The location is not supported. Make sure you didn't add a new TensorLocation and haven't updated this code appropriately");
                    }
                }
                finally
                {
                    this.Location = TensorLocation.Cpu;
                }
            }
        }
    }
}
