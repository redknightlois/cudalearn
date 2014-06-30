#include "common.h"

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
}




