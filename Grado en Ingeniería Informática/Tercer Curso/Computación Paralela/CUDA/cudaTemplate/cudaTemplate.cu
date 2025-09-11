%%writefile cudaTemplate.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define CT_MEM_SIZE 8
#define SH_MEM_SIZE 1024

const float const_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

// Array in constant memory
__constant__ float const_d[CT_MEM_SIZE];

/*
 * CUDA Kernel for Device Functionality
 *
 */
__global__ void foo(float *g_data)
{
	// shared memory
	__shared__ float s_data[SH_MEM_SIZE];
	
	// number of threads in the block
	int blockSize = blockDim.x * blockDim.y;	// COMPLETAR...
	
	// local thread ID (in thread block)
	int tid_b = threadIdx.y * blockDim.x + threadIdx.x;		// COMPLETAR...
	
	// global thread ID (in grid)
	int tid_g = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
	blockIdx.x * blockDim.x * blockDim.y +
	threadIdx.y * blockDim.x + threadIdx.x; 	// COMPLETAR...
	
	// copy data from global memory to shared memory
	s_data[tid_b] = g_data[tid_g];// COMPLETAR...
	__syncthreads();
	
	// perform some computations in shared memory
	s_data[tid_b] = (float) (tid_g + const_d[tid_g % CT_MEM_SIZE]);
	__syncthreads();
	
	// copy data from shared memory to global memory
	// COMPLETAR...
	g_data[tid_g] = s_data[tid_b];
}

/*
 * Host Main Routine
 *
 */
int main(int argc, char **argv)
{
	int dimGrid_x, dimGrid_y;
	int dimBlock_x, dimBlock_y;
	
	// default values for input data
	dimGrid_x = dimGrid_y = 2;
	dimBlock_x = dimBlock_y = 16;
	
	// check block size for use of shared memory in GPU kernel
	size_t block_size = dimBlock_x * dimBlock_y * sizeof(float);
	assert(block_size <= SH_MEM_SIZE * sizeof(float));
	
	int nPos = dimGrid_x * dimGrid_y * dimBlock_x * dimBlock_y;
	size_t nBytes = nPos * sizeof(float);
	
	// allocate host memory
	float *h_data = (float *) malloc(nBytes);
	bzero(h_data, nBytes);
	
	// Set the GPU to use
	// COMPLETAR...
	cudaSetDevice(0);

	
	float *d_data = NULL;

	// allocate device memory
	// COMPLETAR...
	cudaMalloc((void **)&d_data, nBytes);

	
	// copy data from host memory to device memory
	// COMPLETAR...
	cudaMemcpy(d_data, h_data, nBytes, cudaMemcpyHostToDevice);

	
	// initialize constant memory
	cudaMemcpyToSymbol(const_d, const_h, CT_MEM_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
	
	// setup grid and thread block
	dim3 grid(dimGrid_x, dimGrid_y);
	dim3 block(dimBlock_x, dimBlock_y);
	
	// launch the kernel
	foo<<<grid, block>>>(d_data);
	
	// check if kernel execution generated and error
	// COMPLETAR...
	cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    return EXIT_FAILURE;
}

	
	// copy results from device memory to host memory
	// COMPLETAR...
	cudaMemcpy(h_data, d_data, nBytes, cudaMemcpyDeviceToHost);


	// check results
	for(int i = 0; i < nPos; i++) {
		assert(h_data[i] == (i + const_h[i % CT_MEM_SIZE]));
	}
	
	// free device memory
	// COMPLETAR...
	cudaFree(d_data);

	
	// free host memory
	free(h_data);
	
	printf("\nTEST PASSED!\n\n");
	return(EXIT_SUCCESS);
}
