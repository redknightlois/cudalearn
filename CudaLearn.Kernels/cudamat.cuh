#ifndef NVMATRIX_KERNEL_H_
#define NVMATRIX_KERNEL_H_

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

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
* Defines for getting the values at the lower and upper 32 bits
* of a 64-bit number.
*/
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

/*
* Number of iterations to run random number generator upon initialization.
*/
#define NUM_RND_BURNIN                      100

/*
* CUDA grid dimensions for different types of kernels
*/
#define COPY_BLOCK_SIZE                     16

// element-wise kernels use min(ceil(N / 512), 4096) blocks of 512 threads
#define MAX_VECTOR_OP_BLOCKS                4096
#define MAX_VECTOR_OP_THREADS_PER_BLOCK     512
#define NUM_VECTOR_OP_BLOCKS(N)             (min(((N) + MAX_VECTOR_OP_THREADS_PER_BLOCK - 1)/MAX_VECTOR_OP_THREADS_PER_BLOCK, MAX_VECTOR_OP_BLOCKS))
#define NUM_VECTOR_OP_THREADS_PER_BLOCK(N)  (min((N), MAX_VECTOR_OP_THREADS_PER_BLOCK))

#define PI 3.1415926535897932f

#endif