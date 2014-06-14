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




template< typename T >
__device__ void vectorEquals(const T *A, const T *B, bool *C, int numElements, T epsilon)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	// A global result.
	if (i == 0)
		C[0] = true;

	__syncthreads();

	if (i < numElements)
	{
		// Calculate the difference. 
		T result = A[i] - B[i];
		result = result >= 0 ? result : -result;

		// We dont care who wins. We will only signal C[0] when this value changes. So there is no need for atomics or handling the race condition.
		if (result >= epsilon)
			C[0] = false;
	}
}

/**
* Vector addition: C = A + B.
*
* This sample is a very basic sample that implements element by element
* vector addition. It is the same as the sample illustrating Chapter 2
* of the programming guide with some additions like error checking.
*/
template< typename T >
__device__ void vectorAdd(const T *A, const T *B, T *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

template< typename T >
__device__ void vectorAxpby(const T *x, const T a, const T *y, const T b, T *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = a * x[i] + b * y[i];
	}
}

template< typename T >
__device__ void vectorAxpb(const T *x, const T a, const T b, T *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = a * x[i] + b;
	}
}

extern "C"  
{

	__global__ void vectorAdd1f(const float *A, const float *B, float *C, int numElements)
	{
		vectorAdd<float>(A, B, C, numElements);
	}

	__global__ void vectorAdd1d(const double *A, const double *B, double *C, int numElements)
	{
		vectorAdd<double>(A, B, C, numElements);
	}

	__global__ void vectorAdd1i(const int *A, const int *B, int *C, int numElements)
	{
		vectorAdd<int>(A, B, C, numElements);
	}


	__global__ void vectorAxpby1f(const float *x, const float a, const float *y, const float b, float *C, int numElements)
	{
		vectorAxpby<float>(x, a, y, b, C, numElements);
	}

	__global__ void vectorAxpby1d(const double *x, const double a, const double *y, const double b, double *C, int numElements)
	{
		vectorAxpby<double>(x, a, y, b, C, numElements);
	}

	__global__ void vectorAxpby1i(const int *x, const int a, const int *y, const int b, int *C, int numElements)
	{
		vectorAxpby<int>(x, a, y, b, C, numElements);
	}


	__global__ void vectorAxpb1f(const float *x, const float a, const float b, float *C, int numElements)
	{
		vectorAxpb<float>(x, a, b, C, numElements);
	}

	__global__ void vectorAxpb1d(const double *x, const double a, const double b, double *C, int numElements)
	{
		vectorAxpb<double>(x, a, b, C, numElements);
	}

	__global__ void vectorAxpb1i(const int *x, const int a, const int b, int *C, int numElements)
	{
		vectorAxpb<int>(x, a, b, C, numElements);
	}


	__global__ void vectorEquals1f(const float *A, const float *B, bool *C, int numElements, float epsilon)
	{
		vectorEquals<float>(A, B, C, numElements, epsilon);
	}

	__global__ void vectorEquals1d(const double *A, const double *B, bool *C, int numElements, double epsilon)
	{
		vectorEquals<double>(A, B, C, numElements, epsilon);
	}

	__global__ void vectorEquals1i(const int *A, const int *B, bool *C, int numElements, int epsilon)
	{
		vectorEquals<int>(A, B, C, numElements, epsilon);
	}
}




