#include "common.h"

template< typename T >
__device__ void matrixSetIdentity(T *a, const unsigned int rows)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Matrix is square to have an identity
	if (y < rows && x < rows)
	{
		const int p = rows * y + x;
		if (x == y)
			a[p] = 1;
		else
			a[p] = 0;
	}
}

template< typename T >
__device__ void matrixAddColVector(const T* mat, const T* vec, T alpha, T* target, const unsigned int width, const unsigned int height)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < width * height; i += numThreads) 
	{
		target[i] = mat[i] + alpha * vec[i / height];
	}
}

template< typename T >
__device__ void matrixAddRowVector(const T* mat, const T* vec, T alpha, T* target, const unsigned int width, const unsigned int height)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < width * height; i += numThreads) 
	{
		target[i] = mat[i] + alpha * vec[i % height];
	}
}


extern "C"
{

	__global__ void matrixSetIdentity1f(float *x, const unsigned int rows)
	{
		matrixSetIdentity<float>(x, rows);
	}

	__global__ void matrixSetIdentity1d(double *x, const unsigned int rows)
	{
		matrixSetIdentity<double>(x, rows);
	}

	__global__ void matrixSetIdentity1i(int *x, const unsigned int rows)
	{
		matrixSetIdentity<int>(x, rows);
	}

	__global__ void matrixAddColVector1f(const float* mat, const float* vec, float alpha, float* target, const unsigned int width, const unsigned int height)
	{
		matrixAddColVector<float>(mat, vec, alpha, target, width, height);
	}

	__global__ void matrixAddColVector1d(const double* mat, const double* vec, double alpha, double* target, const unsigned int width, const unsigned int height)
	{
		matrixAddColVector<double>(mat, vec, alpha, target, width, height);
	}

	__global__ void matrixAddColVector1i(const int* mat, const int* vec, int alpha, int* target, const unsigned int width, const unsigned int height)
	{
		matrixAddColVector<int>(mat, vec, alpha, target, width, height);
	}

	__global__ void matrixAddRowVector1f(const float* mat, const float* vec, float alpha, float* target, const unsigned int width, const unsigned int height)
	{
		matrixAddRowVector<float>(mat, vec, alpha, target, width, height);
	}

	__global__ void matrixAddRowVector1d(const double* mat, const double* vec, double alpha, double* target, const unsigned int width, const unsigned int height)
	{
		matrixAddRowVector<double>(mat, vec, alpha, target, width, height);
	}

	__global__ void matrixAddRowVector1i(const int* mat, const int* vec, int alpha, int* target, const unsigned int width, const unsigned int height)
	{
		matrixAddRowVector<int>(mat, vec, alpha, target, width, height);
	}

}