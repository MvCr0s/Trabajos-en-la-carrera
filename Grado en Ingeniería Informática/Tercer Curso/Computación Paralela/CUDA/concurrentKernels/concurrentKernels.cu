%%writefile concurrentKernels.cu

/*
 * This sample demonstrates the use of streams for concurrent execution
 * and illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.
 * 
 * Events are inserted into a stream of CUDA calls. Since CUDA streams
 * are asynchronous, the CPU can perform computations while GPU is
 * executing (including DMA memcopies between the host and device)
 *
 */

// Includes C
#include <stdio.h>

// This kernel runs at least for a specified number of clocks
__global__ void clock_block(clock_t *d_t, clock_t clock_count)

{
	clock_t clock_offset = 0;
	clock_t start_clock = clock();
	
	while (clock_offset < clock_count) {
		clock_offset = (clock_t) (clock() - start_clock);
	}
	d_t[0] = clock_offset;
}

/*
 * Host main routine
 *
 */
int main(int argc, char **argv)
{
	unsigned long int counter = 0;
	int nkernels, nstreams;
	float kernel_time = 10;		// time the kernel should run in ms
	float elapsed_time;			// timing variables
	
	nkernels = 8;				// default number of concurrent kernels
	nstreams = nkernels + 1;	// default number of streams
	
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	
	if ((deviceProp.concurrentKernels == 0)) {
		printf("> GPU does not support concurrent kernel execution\n");
		printf("  CUDA kernel runs will be serialized\n");
	}
	
	// number of bytes required to allocate host memory
	int nbytes = nkernels * sizeof(clock_t);
	
	// pointer to data in host memory
	clock_t *h_data = NULL;
	
	// allocate pinned host memory
	// COMPLETAR...
	
	// pointer to data in the device memory
	clock_t *d_data;
	
	// allocate device memory
	cudaMalloc((void **)&d_data, nbytes);
	
	// allocate and initialize an array of stream handles
	cudaStream_t *streams = // COMPLETAR...
	
	// create streams
	for (int i = 0; i < nstreams; i++) {
		// COMPLETAR...
	}
	
	// array of events to handle each concurrent kernel
	cudaEvent_t *kernelEvent = (cudaEvent_t *) malloc(nkernels * sizeof(cudaEvent_t));
	
	for (int i = 0; i < nkernels; i++) {
		// this kernel events are used for synchronization only
		cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming);
	}
	
	// define event handles
	cudaEvent_t start, stop;
	
	// create events
	// COMPLETAR...
	
	// time clocks for each kernel
	clock_t total_clocks = 0;
	clock_t time_clocks  = (clock_t) (kernel_time * deviceProp.clockRate);
	
	// insert stream 0 in start event
	// COMPLETAR...
	
	// queue kernels in separate streams
	for (int i = 0; i < nkernels; ++i) {
		clock_block<<< 1, 1, // COMPLETAR... >>>(&d_data[i], time_clocks);
		total_clocks += time_clocks;
		
		// insert event for each concurrent kernel
		cudaEventRecord(kernelEvent[i], streams[i]);
		
		// make the last stream wait for the kernel event
		cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[i], 0);
	}
	
	// copy back to host in asynchronous mode using the last CUDA stream
	// COMPLETAR...
	
	// insert stream 0 in stop event
	// COMPLETAR...
	
	// at this point the CPU has dispatched all work for the GPU and can continue processing other tasks in parallel
	while (cudaEventQuery(stop) == cudaErrorNotReady) { counter++; }
	
	// just wait until the GPU is done
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	
	// print the GPU times
	printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels, nkernels * kernel_time / 1000.0f);
	printf("Expected time for concurrent execution of %d kernels = %.3fs\n", nkernels, kernel_time / 1000.0f);
	printf("Measured time for GPU = %.3fs\n", elapsed_time / 1000.0f);
	
	// check the output for correctness
	for (int i = 1; i < nkernels; ++i) {
		h_data[0] += h_data[i];
	}
	bool bTestResult = (h_data[0] > total_clocks);
	printf("CPU executed %lu iterations while waiting for GPU to finish\n\n", counter);
	
	// destroy events
	for (int i = 0; i < nkernels; i++) {
		// COMPLETAR...
	}
	
	// destroy streams
	for (int i = 0; i < nstreams; i++) {
		// COMPLETAR...
	}
	
	// free array of stream handles
	free(streams);
	free(kernelEvent);
	
	// free device memory
	cudaFree(d_data);
	
	// free pinned host memory
	// COMPLETAR...
	
	if (!bTestResult) {
		printf("Test Failed!\n");
		exit(EXIT_FAILURE);
	}
	
	printf("Test Passed\n");
	exit(EXIT_SUCCESS);
}
