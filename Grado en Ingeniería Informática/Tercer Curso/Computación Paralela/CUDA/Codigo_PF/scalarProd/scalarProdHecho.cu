%%writefile scalarProd.cu

/*
 * This file contains both device and host code to calculate the
 * scalar product of two vectors of N elements.
 * 
 */

#include <stdio.h>

#define N 1024
#define SEGMENT_SIZE 64

///////////////////////////////////////////////////////////////////////////////
//
// Computes the scalar product of two vectors of N elements on GPU.
//
///////////////////////////////////////////////////////////////////////////////
__global__ void scalarProd(float *C, const float *A, const float *B, int nElem) {

	// COMPLETAR...
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < nElem){
        C[tid]=A[tid]*B[tid];
    }
}

/////////////////////////////////////////////////////////////////
//
// Computes a standard parallel reduction on GPU.
//
/////////////////////////////////////////////////////////////////
__global__ void vectorReduce(float *R, const float *C, int nElem)
{
	// Array in Shared Memory
    extern __shared__ float sdata[];
	
	// COMPLETAR...
    int tid = threadIdx.x;

    int tig = blockIdx.x * blockDim.x + tid;

    if(tig < nElem){
        sdata[tid]=C[tig];
    }else{
        sdata[tid]=0.0;
    }

    __syncthreads();

    for(int s= blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid==0){
        R[ blockIdx.x ]=sdata[0];
    }
}

// -----------------------------------------------
// Host Utility Routines
// -----------------------------------------------
float scalarProd_CPU(float *A, float *B, int nElem)
{
	float suma = 0.0f;	
	for (int i = 0; i < nElem; i++) {
		suma += A[i] * B[i];
	}
	return suma;
}

float randFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + (t * high);
}

// ------------
// Main Program
// ------------
int main( void ) {

	// Array Elements
	int n_elem = N;
	
	// Block Dimension
	int block_dim = SEGMENT_SIZE;
	
	// Number of Blocks
	int n_block = ( n_elem % block_dim == 0 ) ? (n_elem / block_dim) : ((n_elem / block_dim) + 1); // COMPLETAR...
	
	// Execution Configuration Parameters
	dim3 blocks ( n_block );// COMPLETAR...
	dim3 threads( block_dim );// COMPLETAR...
	
	// Size (in bytes) Required to Store the Matrix
	size_t n_bytes = (n_elem * sizeof(float));
	
	// Allocate Host Memory
	float *h_A = (float *) malloc( n_bytes );// COMPLETAR...
	float *h_B = (float *) malloc( n_bytes );// COMPLETAR...
	float *h_R = (float *) malloc( n_block * sizeof(float) );// COMPLETAR...
		
	// Initialize Host Data
	srand(123);
	
	// Generating input data on CPU
	for (int i=0; i < n_elem; i++) {
		h_A[i] = randFloat(0.0f, 1.0f);
		h_B[i] = randFloat(0.0f, 1.0f);
	}
	
	// Compute Reference CPU Solution
	float result_cpu = scalarProd_CPU(h_A, h_B, n_elem);
	
	// CUDA Events
	cudaEvent_t start, stop;
	
	// Allocate Device Memory
	float *d_A, *d_B, *d_C, *d_R;
	cudaMalloc((void **)&d_A, n_bytes );// COMPLETAR...
	cudaMalloc((void **)&d_B, n_bytes );// COMPLETAR...
	cudaMalloc((void **)&d_C, n_bytes );// COMPLETAR...
	cudaMalloc((void **)&d_R, n_block * sizeof(float) );// COMPLETAR...
	
	// Init Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop );
	
	// Start Time Measurement
    cudaEventRecord(start, 0);
	
	// Copy Host Data to Device
	
	// COMPLETAR...
    cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice);

	scalarProd<<< n_block, threads >>>(d_C, d_A, d_B, n_elem); // COMPLETAR...
	cudaDeviceSynchronize();
	vectorReduce<<< n_block, threads, block_dim * sizeof(float) >>>(d_R, d_C, n_elem);// COMPLETAR...(teniendo en cuenta memoria shared)
	
	// Copy Device Data to Host
    
	// COMPLETAR...
    cudaMemcpy(h_R, d_R, n_block * sizeof(float), cudaMemcpyDeviceToHost);
	
	// End Time Measurement
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, stop);
    printf("Execution Time by the GPU: %.2f\n", kernel_time);

	float result_gpu = 0.0f;
	for (int i=0; i < n_block; i++) {
		result_gpu += h_R[i];
	}
	
	// Free Host Memory
	free(h_A); free(h_B); free(h_R);
	
	// Free Device Memory
	cudaFree(d_A); cudaFree(d_B);
	cudaFree(d_C); cudaFree(d_R);
	
	// Destroy Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	if (result_cpu != result_cpu) {
		printf("Test Failed!\n");
		exit(EXIT_FAILURE);
	}
	printf("Test Passed\n");
	exit(EXIT_SUCCESS);
}
