%%writefile vectorReduce.cu

/*

* This example shows how to compute the reduction of the elements of a vector.

*

* Also, it shows how to measure the performance of block of threads of a kernel

* accurately. Blocks are executed in parallel and out of order. Since there's no

* synchronization mechanism between blocks, we measure the clock once for each block.

*

*/

#include <stdio.h>

#include <assert.h>

        /*

        * vectorReduce

        *

        * This kernel computes a standard parallel reduction and evaluates the

        * time it takes to do that for each block. The timing results are stored in device memory.

        *

        */

        __global__ void
        time_and_reduce_partial(float *vector_d, float *reduce_d, clock_t *times_d, int n)

{

    extern __shared__ float sdata[];

    // local thread ID (in block)

    int tidb = threadIdx.x;

    // global thread (ID in grid)

    int tidg = blockDim.x * blockIdx.x + threadIdx.x;

    // record the initial time for each block

    if (tidb == 0)
    {

        times_d[blockIdx.x] = clock();
    }

    // move data from global to shared memory

    if (tidg < n)
    {

        sdata[tidb] = vector_d[tidg];
    }

    else
    {

        sdata[tidb] = 0.0f;
    }

    __syncthreads();

    // perform reduction in shared memory

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {

        if (tidb < s)
        {

            sdata[tidb] += sdata[tidb + s];
        }

        __syncthreads();
    }

    // write result for this block to global memory

    if (tidb == 0)
    {

        reduce_d[blockIdx.x] = sdata[0];

        times_d[blockIdx.x] = clock() - times_d[blockIdx.x];
    }
}

__global__ void reduce_final_gpu(float *partial_sums, float *final_result, int num_partial_sums)

{

    extern __shared__ float sdata_final[];

    // local thread ID (in block)

    int tidb = threadIdx.x;

    if (tidb < num_partial_sums)
    {

        sdata_final[tidb] = partial_sums[tidb];
    }

    else
    {

        sdata_final[tidb] = 0.0f;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    { // blockDim.x del kernel reduce_final_gpu

        if (tidb < s)
        {

            sdata_final[tidb] += sdata_final[tidb + s];
        }

        __syncthreads();
    }

    if (tidb == 0)
    {

        *final_result = sdata_final[0];
    }
}

/*

* Host main routine

*

*/

int main(int argc, char **argv)

{

    // default parameter values

    int n = 1024, bsx = 32;

    size_t nBytes = n * sizeof(float);

    clock_t *clocks_h = NULL;

    clock_t *clocks_d = NULL;

    float elapsed_time = .0;

    float *vector_h, *reduce_h; // host data

    float *vector_d, *reduce_d; // device data

    float final_result_h;

    float *final_result_d;

    // set the GPU to use

    cudaSetDevice(0);

    // total number of thread blocks

    int nblocks = (n + bsx - 1) / bsx;

    // set kernel launch configuration

    dim3 grid(nblocks);

    dim3 block(bsx);

    // allocate host memory

    vector_h = (float *)malloc(nBytes);

    clocks_h = (clock_t *)malloc(nblocks * sizeof(clock_t));

    reduce_h = (float *)malloc(nblocks * sizeof(float));

    float acum = .0;

    // initialize host memory

    for (int i = 0; i < n; i++)
    {

        vector_h[i] = (float)1;

        acum += 1.0;
    }

    // allocate device memory

    cudaMalloc((void **)&vector_d, nBytes);

    cudaMalloc((void **)&reduce_d, nblocks * sizeof(float));

    cudaMalloc((void **)&clocks_d, nblocks * sizeof(clock_t));

    cudaMalloc((void **)&final_result_d, sizeof(float));

    // create cuda events

    cudaEvent_t start, stop;

    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    // insert stream 0 in start event

    cudaEventRecord(start, 0);

    // copy data from host memory to device memory

    cudaMemcpy(vector_d, vector_h, nBytes, cudaMemcpyHostToDevice);

    // execute the kernel

    printf("---> Running configuration: grid of %d blocks of %d threads (TOTAL: %d threads)\n", nblocks, bsx, nblocks * bsx);

    time_and_reduce_partial<<<grid, block, bsx * sizeof(float)>>>(vector_d, reduce_d, clocks_d, n);

    dim3 grid_final(1);

    dim3 block_final(256);

    reduce_final_gpu<<<grid_final, block_final, 256 * sizeof(float)>>>(reduce_d, final_result_d, nblocks);

    // copy data from device memory to host memory

    cudaMemcpy(&final_result_h, final_result_d, sizeof(float), cudaMemcpyDeviceToHost);

    // insert stream 0 in stop event

    cudaEventRecord(stop, 0);

    // using events to calculate the execution time

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("---> Time spent executing by the GPU: %.2f\n", elapsed_time);

    long double avgElapsedClocks = 0;

    for (int i = 0; i < nblocks; i++)
    {

        avgElapsedClocks += (long double)clocks_h[i];
    }

    avgElapsedClocks = avgElapsedClocks / nblocks;

    printf("Average Clocks/Block = %Lf\n", avgElapsedClocks);

    // check the output for correctness

    assert(final_result_h == (float)acum);

    // destroy events

    cudaEventDestroy(start);

    cudaEventDestroy(stop);

    // free host memory

    free(vector_h);

    free(reduce_h);

    free(clocks_h);

    // free device memory

    cudaFree((void *)vector_d);

    cudaFree((void *)reduce_d);

    cudaFree((void *)clocks_d);

    cudaFree((void *)final_result_d);

    printf("\nTest PASSED\n");

    exit(EXIT_SUCCESS);
}