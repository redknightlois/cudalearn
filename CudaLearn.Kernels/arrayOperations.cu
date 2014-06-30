#include "common.h"

template< typename T >
__device__ void arrayEquals(const T *A, const T *B, bool *C, int numElements, T epsilon)
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


template< typename T >
__device__ void arraySet(T *x, const T a, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		x[i] = a;
	}
}

template< typename T >
__device__ void arrayBinaryLess(const T* m1, const T* m2, T* target, const unsigned int len)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < len; i += numThreads)
	{
		target[i] = m1[i] < m2[2] ? 1 : 0;
	}
}



extern "C"
{
	__global__ void arraySet1f(float *x, const float a, int numElements)
	{
		arraySet<float>(x, a, numElements);
	}

	__global__ void arraySet1d(double *x, const double a, int numElements)
	{
		arraySet<double>(x, a, numElements);
	}

	__global__ void arraySet1i(int *x, const int a, int numElements)
	{
		arraySet<int>(x, a, numElements);
	}


	__global__ void arrayEquals1f(const float *A, const float *B, bool *C, int numElements, float epsilon)
	{
		arrayEquals<float>(A, B, C, numElements, epsilon);
	}

	__global__ void arrayEquals1d(const double *A, const double *B, bool *C, int numElements, double epsilon)
	{
		arrayEquals<double>(A, B, C, numElements, epsilon);
	}

	__global__ void arrayEquals1i(const int *A, const int *B, bool *C, int numElements, int epsilon)
	{
		arrayEquals<int>(A, B, C, numElements, epsilon);
	}

	__global__ void arrayBinaryLess1f(const float *A, const float *B, float *target, const unsigned int numElements)
	{
		arrayBinaryLess<float>(A, B, target, numElements);
	}

	__global__ void arrayBinaryLess1d(const double *A, const double *B, double *target, const unsigned int numElements)
	{
		arrayBinaryLess<double>(A, B, target, numElements);
	}

	__global__ void arrayBinaryLess1i(const int *A, const int *B, int *target, const unsigned int numElements)
	{
		arrayBinaryLess<int>(A, B, target, numElements);
	}
}