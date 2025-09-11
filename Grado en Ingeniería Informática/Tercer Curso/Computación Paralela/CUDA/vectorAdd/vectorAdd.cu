%%writefile vectorAdd.cu

#include <math.h>
#include <stdio.h>

/*
 * CUDA Kernel Device Code
 *
 */
__global__ void vectorAdd( const float *A, const float *B, float *S, int nElem ) {

}

/*
 * Host Main Routine
 *
 */
int main( int argc, char *argv[] )
{
  int i;
  int nBlock = 1;
  int bSize  = 128;
  int nElem  = 10000;
  
  size_t size = nElem * sizeof(float);

  // Allocate the host input vector A
  float *h_A = (float *) malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *) malloc(size);

  // Allocate the host output vector S
  float *h_S = (float *) malloc(size);
  
  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_S == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (i = 0; i < nElem; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }
  
  // Set the GPU
  cudaSetDevice(0);
  
  // Allocate the device input vector A
  float *d_A = NULL;
  cudaMalloc((void **)&d_A, size);

  // Allocate the device input vector B
  float *d_B = NULL;
  cudaMalloc((void **)&d_B, size);

  // Allocate the device output vector S
  float *d_S = NULL;
  cudaMalloc((void **)&d_S, size);
  
  // Copy the host input vectors A and B to the device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  
  printf("[Vector Addition of %d Elements]\n", nElem);
  
  // Define the Grid and Block dimensions
  dim3 myGrid(nBlock);
  dim3 myBlock(nElem);

  // Launch the Vector Add CUDA Kernel
  printf("CUDA kernel launch with %d blocks of %d threads\n", nBlock, nElem);
  vectorAdd<<<myGrid, myBlock>>>(d_A, d_B, d_S, nElem);
    
  // Copy the device result vector S to the host memory
  cudaMemcpy(h_S, d_S, size, cudaMemcpyDeviceToHost);
  
  // Verify that the resulting vector is correct
  for (i = 0; i < nElem; ++i) {
    if (fabs((h_A[i] + h_B[i]) - h_S[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }
  printf("Test PASSED\n");
  
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_S);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_S);

  printf("Done\n");
  return (EXIT_SUCCESS);
}
