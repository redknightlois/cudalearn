#include "common.h"

template< typename T >
__device__ void mExp(const T* mat, T* target, const unsigned int len)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < len; i += numThreads)
	{
		target[i] = __expf(mat[i]);
	}
}


template< typename T >
__device__ void mPow(const T* mat, T y, T* target, const unsigned int len)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < len; i += numThreads)
	{
		target[i] = powf(mat[i], y);
	}
}


extern "C"
{
	__global__ void mExp1f(const float* mat, float* target, const unsigned int len)
	{
		mExp<float>(mat, target, len);
	}

	__global__ void mExp1d(const double* mat, double* target, const unsigned int len)
	{
		mExp<double>(mat, target, len);
	}

	__global__ void mExp1i(const int* mat, int* target, const unsigned int len)
	{
		mExp<int>(mat, target, len);
	}

	__global__ void mPow1f(const float* mat, float power, float* target, const unsigned int len)
	{
		mPow<float>(mat, power, target, len);
	}

	__global__ void mPow1d(const double* mat, double power, double* target, const unsigned int len)
	{
		mPow<double>(mat, power, target, len);
	}

	__global__ void mPow1i(const int* mat, int power, int* target, const unsigned int len)
	{
		mPow<int>(mat, power, target, len);
	}

}