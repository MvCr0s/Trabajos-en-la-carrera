%%writefile deviceQuery.cu

#include <iostream>
#include <memory>
#include <string>

int   *pArgc = NULL;
char **pArgv = NULL;

int main(int argc, char **argv) {

    pArgc = &argc;
    pArgv = argv;

    printf("%s Starting...\n\n", argv[0]);
    printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s):\n  ", deviceCount);
    }
    printf("\n");

    // This will pick the best possible CUDA capable device
    int dev = 0, driverVersion = 0, runtimeVersion = 0;

    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("  Device %d: \"%s\"\n", dev, deviceProp.name);

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    printf("  Multiprocessors:                               %d\n", deviceProp.multiProcessorCount);

    int cores = 0;
    int mproc = deviceProp.multiProcessorCount;

    switch (deviceProp.major){
        case 2: // Fermi
                if (deviceProp.minor == 1) cores = mproc * 48;
                else cores = mproc * 32;
                break;
        case 3: // Kepler
                cores = mproc * 192;
                break;
        case 5: // Maxwell
                cores = mproc * 128;
                break;
        case 6: // Pascal
                if ((deviceProp.minor == 1) || (deviceProp.minor == 2)) cores = mproc * 128;
                else if (deviceProp.minor == 0) cores = mproc * 64;
                else printf("Unknown device type\n");
                break;
        case 7: // Volta and Turing
                if ((deviceProp.minor == 0) || (deviceProp.minor == 5)) cores = mproc * 64;
                else printf("Unknown device type\n");
                break;
        case 8: // Ampere
                if (deviceProp.minor == 0) cores = mproc * 64;
                else if (deviceProp.minor == 6) cores = mproc * 128;
                else if (deviceProp.minor == 9) cores = mproc * 128; // ada lovelace
                else printf("Unknown device type\n");
                break;
        case 9: // Hopper
                if (deviceProp.minor == 0) cores = mproc * 128;
                else printf("Unknown device type\n");
                break;
        default:
                printf("Unknown device type\n"); 
                break;
    }
    printf("  Number of CUDA cores per multiprocessor:       %d\n", cores);

    printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);

    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

    char msg[256];
    snprintf(msg, sizeof(msg), "  Total amount of global memory:                 %llu bytes\n", (unsigned long long)deviceProp.totalGlobalMem);
    printf("%s", msg);

    printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);

    // cudaDeviceReset causes the driver to clean up all state.
    cudaDeviceReset();

    // finish
    exit(EXIT_SUCCESS);
}
