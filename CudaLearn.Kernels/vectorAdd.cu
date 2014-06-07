#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include "float.h" 

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

extern "C"  
{

	/**
	 * CUDA Kernel Device code
	 *
	 * Computes the vector addition of A and B into C. The 3 vectors have the same
	 * number of elements numElements.
	 */
	__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < numElements)
		{
			C[i] = A[i] + B[i];
		}
	}

}


