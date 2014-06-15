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

}