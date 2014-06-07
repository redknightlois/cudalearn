ManagedCuda 6.0 contains the following libraries:

ManagedCuda.dll:
As the main library in managedCuda, this dll contains the CUDA Driver API wrapper and all wrapper 
classes of this API. ManagedCUDA.dll is compiled as “Any CPU” and can run on 32 and 64-Bit platforms.

CudaBitmapSource.dll:
CudaBitmapSource.dll is a simple try to handle CUDA device memory as a BitmapSource in WPF. It is more 
like a proof of concept than a ready to use library, especially the fact that BitmapSource is a sealed 
class makes a proper implementation difficult. If you have ideas for improvement or a better design, 
please let me know ;-)

CudaBlas.dll:
Wrapper for “cublas32_60.dll” in 32-bit version or “cublas64_60.dll” in 64-bit.

CudaFFT.dll: 
Wrapper for “cufft32_60.dll” in 32-bit version or “cufft64_60.dll” in 64-bit.

CudaRand.dll: 
Wrapper for “curand32_60.dll” in 32-bit version or “curand64_60.dll” in 64-bit.

CudaSparse.dll: 
Wrapper for “cusparse32_60.dll” in 32-bit version or “curand64_60.dll” in 64-bit.

NPP.dll: 
Wrapper for “npp[c,i,s]32_60.dll” in 32-bit version or “npp[c,i,s]64_60.dll” in 64-bit.
The image part of NPP has also image classes for all supported image data formats giving class/method 
based access to the NPP API. An Image in NPPi can specify a ROI (region of interest) where NPP methods 
perform work, necessary for example for kernel based filters to exclude border pixels and can 
directly copy to or from a standard Winforms Bitmap. A NPPImage_"Type" can be converted to 
CudaPitchedDeviceVariable<Type> and vice versa.
The signal part of NPP is implemented as extension methods for CudaDeviceVariable in namespace 
ManagedCuda.NPP.NPPsExtensions. Add a reference to the NPP library and include the 
ManagedCuda.NPP.NPPsExtensions namespace in order to use NPPs primitives.
