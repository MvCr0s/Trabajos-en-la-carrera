%%writefile matrixMul.cu

/**
 * Matrix Multiplication: C = A * B.
 *
 * This file contains both device and host code to compute a matrix multiplication.
 *
 */

#include <math.h>
#include <stdio.h>

#define MATRIX_DIM   32
#define SEGMENT_SIZE 32

// --------------------
// Device Kernels
// --------------------
__global__ void transposeMatrix(float *d_data, int mat_dim) {

	// Array in Shared Memory
	extern __shared__ float sdata[];

  int tid_b = threadIdx.x;
	int tid_g = (blockIdx.y * blockDim.x + tid_b) * mat_dim + (blockIdx.x * blockDim.x);

	for (int i=0; i < blockDim.x; i++) {
		sdata[tid_b * blockDim.x + i] = d_data[tid_g + i];
	}

	__syncthreads();

	tid_b = threadIdx.x;
	tid_g = (blockIdx.x * blockDim.x + tid_b) * mat_dim + (blockIdx.y * blockDim.x);

	for (int i=0; i < blockDim.x; i++) {
		d_data[tid_g + i] = sdata[i * blockDim.x + tid_b];
	}

}

__global__ void scalarProd(float *C, const float *A, const float *B, int nElem) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nElem) {
		C[idx] = A[idx] * B[idx];
	}

}

__global__ void vectorReduce(float *R, const float *C, int nElem)
{
	// Array in Shared Memory
    extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = (idx < nElem) ? C[idx] : 0.0f;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		R[blockIdx.x]=sdata[0];
	}

}

// ---------------------
// Host Utility Routines
// ---------------------
void matrixMul(const float *A, const float *B, float *C, const int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float acum = 0.0f;
			for (int k = 0; k < n; k++) {
				acum += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = acum;
		}
	}
}

bool compareData(float *h_C, float *d_C, int n) {
	float eps = 1.E-3f;
	for (int i = 0; i < n * n; i++) {
		if (fabsf(h_C[i] - d_C[i]) > eps) {
            printf("Error en el índice %d: Host=%.6f, Device=%.6f\n", i, h_C[i], d_C[i]);
			return false;
		}
	}
	return true;
}

void printMatrix(const float *M, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            // Print as integer-like for clarity
            printf("%4.0f ", M[i * dim + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void transposeCPU(float *At, float *A, const int dim_x, const int dim_y)
{
    for (int y = 0; y < dim_y; y++) {
        for (int x = 0; x < dim_x; x++) {
            At[x * dim_y + y] = A[y * dim_x + x];
        }
    }
}



float randFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + (t * high);
}

// ------------
// Main Program
// ------------
int main( void ) {

	// Matrix Dimensions
	int dim_x = MATRIX_DIM;
	int dim_y = dim_x;

	// Matrix Size
	int mat_size = dim_x * dim_y;

	// Block Dimension
	int block_dim = SEGMENT_SIZE;

	// Number of Blocks
	int n_block = ( dim_x % block_dim == 0 ) ? (dim_x / block_dim) : (dim_x / block_dim + 1);

	// Execution Configuration Parameters
	dim3 blocksPerGrid (n_block , n_block);
	dim3 blockPerGridReduce  ( n_block );
	dim3 threadsPerBlock( block_dim );

	// Size Required to Store the Matrix
	size_t n_bytes = (mat_size * sizeof(float));

	// Allocate Pinned Host Memory
	float *h_A, *h_B, *h_C, *h_R;

	cudaHostAlloc((void**)&h_A, n_bytes, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_B, n_bytes, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_C, n_bytes, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_R, n_bytes, cudaHostAllocDefault);

	// Initialize Host Data
	srand(123);

	// Generating input data on CPU
	for (int i=0; i < mat_size; i++) {
		h_A[i] = randFloat(0.0f, 1.0f);
		h_B[i] = randFloat(0.0f, 1.0f);
	}

	// Compute Reference Matrix Multiplication
	matrixMul(h_A, h_B, h_C, dim_x);

	// CUDA Streams
	cudaStream_t stream;

	// Create Stream
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	// Performance Data
	float kernel_time, kernel_bandwidth;

	// Allocate Device Memory
	float *d_A, *d_B, *d_C,*d_Temp;

	cudaMalloc((void **) &d_A, n_bytes);
  cudaMalloc((void **) &d_B, n_bytes);
  cudaMalloc((void **) &d_C, n_bytes);
  cudaMalloc((void **) &d_Temp,dim_x * sizeof(float));

	// CUDA Events
	cudaEvent_t start, stop;

	// Init Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop );

	// Start Time Measurement
    cudaEventRecord(start, stream);

	// Copy Host Data to Device

	cudaMemcpyAsync(d_A,h_A,n_bytes,cudaMemcpyHostToDevice,stream);
  cudaMemcpyAsync(d_B,h_B,n_bytes,cudaMemcpyHostToDevice,stream);

	cudaStreamSynchronize(stream);
	size_t shared_bytes = block_dim * block_dim* sizeof(float);
	transposeMatrix<<< blocksPerGrid , threadsPerBlock , shared_bytes, stream >>>( d_B, dim_x );

	cudaStreamSynchronize(stream);

  for(int i = 0; i < dim_y; i++) {
		for(int j = 0; j < dim_x; j++) {
			scalarProd<<< blockPerGridReduce , threadsPerBlock ,0,stream>>> ( d_Temp, d_A + (i * dim_x) , d_B + (j * dim_x), dim_x );
			cudaStreamSynchronize(stream);
			size_t shared_bytes_reduce = block_dim * sizeof(float);
			vectorReduce<<< blockPerGridReduce , threadsPerBlock , shared_bytes_reduce, stream>>>( d_C + (i * dim_x + j) , d_Temp, dim_x );
		}
	}
	cudaDeviceSynchronize();

	// Copy Device Data to Host

	cudaMemcpyAsync(h_R,d_C,n_bytes,cudaMemcpyDeviceToHost,stream);

	cudaStreamSynchronize(stream);

	// End Time Measurement
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&kernel_time, start, stop);

	bool res = compareData(h_C, h_R, dim_x);

	if (res == true) {
		// Report Effective Bandwidth
		kernel_bandwidth = (2.0f * 1000.0f * n_bytes)/(1024 * 1024 * 1024);
		kernel_bandwidth /= kernel_time;

		printf( "Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, \n",
				 kernel_bandwidth, kernel_time, (dim_x * dim_y) );
	}

	// Free Host Memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFreeHost(h_R);

	// Free Device Memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Destroy Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Destroy Stream
	cudaStreamDestroy(stream);

	if (res == false) {
		printf("Test Failed!\n");
		exit(EXIT_FAILURE);
	}
	printf("Test Passed\n");
	exit(EXIT_SUCCESS);
}