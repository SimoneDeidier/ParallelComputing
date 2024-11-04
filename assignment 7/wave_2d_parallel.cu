#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

// TASK: T1
// Include the cooperative groups library
// BEGIN: T1
#include <cooperative_groups.h>
// END: T1


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// TASK: T1b
// Variables needed for implementation
// BEGIN: T1b

// Simulation parameters: size, step count, and how often to save the state
int_t N = 128, M = 128, max_iteration = 1000000, snapshot_freq = 1000;

// Wave equation parameters, time step is derived from the space step
const real_t c  = 1.0, dx = 1.0, dy = 1.0;
real_t dt;

// Host variables
real_t* h_buffers[3] = {NULL, NULL, NULL};

// Device variables
real_t* d_buffers[3] = {NULL, NULL, NULL};

#define U_prv(i,j) h_buffers[0][((i) + 1) * (N + 2) + (j) + 1]
#define U(i,j)     h_buffers[1][((i) + 1) * (N + 2) + (j) + 1]
#define U_nxt(i,j) h_buffers[2][((i) + 1) * (N + 2) + (j) + 1]
// END: T1b

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}


// Rotate the time step buffers.
void move_buffer_window(void) {
    real_t* temp = d_buffers[0];
    d_buffers[0] = d_buffers[1];
    d_buffers[1] = d_buffers[2];
    d_buffers[2] = temp;
}


// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    FILE* out = fopen(filename, "wb");
    for(int_t i = 0; i < M; i++) {
        fwrite(&U(i,0), sizeof(real_t), N, out );
    }
    fclose(out);
}


// TASK: T4
// Get rid of all the memory allocations
void domain_finalize(void) {
// BEGIN: T4
    // Free host memory
    free(h_buffers[0]);
    free(h_buffers[1]);
    free(h_buffers[2]);

    // Free device memory
    cudaErrorCheck(cudaFree(d_buffers[0]));
    cudaErrorCheck(cudaFree(d_buffers[1]));
    cudaErrorCheck(cudaFree(d_buffers[2]));
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition
// BEGIN: T6
// Kernel to apply Neumann (reflective) boundary conditions
__global__ void boundary_condition_kernel(real_t* d_buffers, int_t N, int_t M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Apply boundary conditions along the vertical edges
    if (idx < M) {
        d_buffers[(idx + 1) * (N + 2)] = d_buffers[(idx + 1) * (N + 2) + 2];
        d_buffers[(idx + 1) * (N + 2) + (N + 1)] = d_buffers[(idx + 1) * (N + 2) + (N - 1)];
    }

    // Apply boundary conditions along the horizontal edges
    if (idx < N) {
        d_buffers[idx + 1] = d_buffers[(N + 2) + (idx + 1)];
        d_buffers[(M + 1) * (N + 2) + (idx + 1)] = d_buffers[(M - 1) * (N + 2) + (idx + 1)];
    }
}

// Function to launch the boundary condition kernel
void boundary_condition(void) {
    int threadsPerBlock = 256;
    int blocksPerGridM = (M + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridN = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel for vertical edges
    boundary_condition_kernel<<<blocksPerGridM, threadsPerBlock>>>(d_buffers[1], N, M);
    // Launch kernel for horizontal edges
    boundary_condition_kernel<<<blocksPerGridN, threadsPerBlock>>>(d_buffers[1], N, M);

    // Check for any errors in kernel launch
    cudaErrorCheck(cudaPeekAtLastError());
    // Synchronize device to ensure completion
    cudaErrorCheck(cudaDeviceSynchronize());
}
// END: T6


// TASK: T5
// Integration formula
// BEGIN; T5
// Kernel function to compute the next time step of the wave equation
__global__ void time_step_kernel(real_t* d_buffers0, real_t* d_buffers1, real_t* d_buffers2, int_t N, int_t M, real_t dt, real_t c, real_t dx, real_t dy) {
    // Calculate the global row and column indices
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the indices are within the bounds of the grid
    if(i < M && j < N) {
        // Compute the next time step using the finite difference method
        d_buffers2[(i + 1) * (N + 2) + (j + 1)] = -d_buffers0[(i + 1) * (N + 2) + (j + 1)] 
            + 2.0 * d_buffers1[(i + 1) * (N + 2) + (j + 1)] 
            + (dt * dt * c * c) / (dx * dy) * (d_buffers1[(i) * (N + 2) + (j + 1)] 
            + d_buffers1[(i + 2) * (N + 2) + (j + 1)] 
            + d_buffers1[(i + 1) * (N + 2) + (j)] 
            + d_buffers1[(i + 1) * (N + 2) + (j + 2)] 
            - 4.0 * d_buffers1[(i + 1) * (N + 2) + (j + 1)]);
    }
}

// Function to launch the time step kernel
void time_step(void) {
    // Define the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel to compute the next time step
    time_step_kernel<<<numBlocks, threadsPerBlock>>>(d_buffers[0], d_buffers[1], d_buffers[2], N, M, dt, c, dx, dy);
    
    // Check for any errors in kernel launch
    cudaErrorCheck(cudaPeekAtLastError());
    
    // Synchronize device to ensure completion
    cudaErrorCheck(cudaDeviceSynchronize());
}
// END: T5


// TASK: T7
// Main time integration.
void simulate(void) {
// BEGIN: T7
    // Define the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Go through each time step
    for(int_t iteration = 0; iteration <= max_iteration; iteration++) {
        if((iteration % snapshot_freq) == 0) {
            // Copy the device buffer to the host buffer
            cudaErrorCheck(cudaMemcpy(h_buffers[1], d_buffers[1], (M + 2) * (N + 2) * sizeof(real_t), cudaMemcpyDeviceToHost));
            domain_save(iteration / snapshot_freq);
        }

        // Derive step t+1 from steps t and t-1
        boundary_condition();
        time_step_kernel<<<numBlocks, threadsPerBlock>>>(d_buffers[0], d_buffers[1], d_buffers[2], N, M, dt, c, dx, dy);
        cudaErrorCheck(cudaPeekAtLastError());
        cudaErrorCheck(cudaDeviceSynchronize());

        // Rotate the time step buffers
        move_buffer_window();
    }
// END: T7
}


// TASK: T8
// GPU occupancy
void occupancy(void) {
// BEGIN: T8
    // Get device properties for device 0
    cudaDeviceProp prop;
    cudaErrorCheck(cudaGetDeviceProperties(&prop, 0));

    // Maximum number of threads per block
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    // Maximum number of blocks per streaming multiprocessor
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / maxThreadsPerBlock;
    // Number of streaming multiprocessors
    int numSMs = prop.multiProcessorCount;

    // Calculate the maximum number of active blocks per multiprocessor
    int maxActiveBlocks;
    cudaErrorCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, time_step_kernel, maxThreadsPerBlock, 0));

    // Calculate the theoretical occupancy
    float occupancy = (float)(maxActiveBlocks * maxThreadsPerBlock) / (float)(maxBlocksPerSM * numSMs * maxThreadsPerBlock);
    printf("Grid size set to %d x %d\n", maxBlocksPerSM, maxThreadsPerBlock);
    printf("Launched block of size %d x %d\n", numSMs, maxActiveBlocks);
    printf("Theoretical occupancy: %f\n", occupancy);
// END: T8
}


// TASK: T2
// Make sure at least one CUDA-capable device exists
static bool init_cuda() {
    // BEGIN: T2
    int device_count;
    // Get the number of CUDA-capable devices
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if(err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA-compatible device found.\n");
        return false;
    }

    printf("CUDA device count: %d\n", device_count);

    // Iterate through each device and print its properties
    for(int device = 0; device < device_count; ++device) {
        cudaDeviceProp device_prop;
        err = cudaGetDeviceProperties(&device_prop, device);
        if(err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", device, cudaGetErrorString(err));
            return false;
        }

        printf("CUDA device #%d\n", device);
        printf("\tName: %s\n", device_prop.name);
        printf("\tCompute capability: %d.%d\n", device_prop.major, device_prop.minor);
        printf("\tMultiprocessors: %d\n", device_prop.multiProcessorCount);
        printf("\tWarp size: %d\n", device_prop.warpSize);
        printf("\tGlobal memory: %zu bytes\n", device_prop.totalGlobalMem);
        printf("\tPer-block shared memory: %zu bytes\n", device_prop.sharedMemPerBlock);
        printf("\tPer-block registers: %d\n", device_prop.regsPerBlock);
    }

    // Set the device to be used for GPU executions
    err = cudaSetDevice(0);
    if(err != cudaSuccess) {
        fprintf(stderr, "Failed to set device 0: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
    // END: T2
}



// TASK: T3
// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize(void) {
// BEGIN: T3
    bool locate_cuda = init_cuda();
    if(!locate_cuda) {
        exit(EXIT_FAILURE);
    }

    // Allocate space for at least one grid on the host
    h_buffers[0] = (real_t*) malloc((M + 2) * (N + 2) * sizeof(real_t));
    h_buffers[1] = (real_t*) malloc((M + 2) * (N + 2) * sizeof(real_t));
    h_buffers[2] = (real_t*) malloc((M + 2) * (N + 2) * sizeof(real_t));

    // Allocate space for three grids on the device
    cudaErrorCheck(cudaMalloc((void**)&d_buffers[0], (M + 2) * (N + 2) * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc((void**)&d_buffers[1], (M + 2) * (N + 2) * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc((void**)&d_buffers[2], (M + 2) * (N + 2) * sizeof(real_t)));

    for(int_t i = 0; i < M; i++) { 
        for(int_t j = 0; j < N; j++) {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt(((i - M/2.0) * (i - M/2.0)) / (real_t)M + ((j - N/2.0) * (j - N/2.0)) / (real_t)N);
            h_buffers[0][((i) + 1) * (N + 2) + (j) + 1] = h_buffers[1][((i) + 1) * (N + 2) + (j) + 1] = exp(-4.0 * delta * delta);
        }
    }

    // Copy host buffers to device buffers
    cudaErrorCheck(cudaMemcpy(d_buffers[0], h_buffers[0], (M + 2) * (N + 2) * sizeof(real_t), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_buffers[1], h_buffers[1], (M + 2) * (N + 2) * sizeof(real_t), cudaMemcpyHostToDevice));

    // Set the time step for 2D case
    dt = dx * dy / (c * sqrt(dx * dx + dy * dy));
// END: T3
}


int main(void) {
    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    gettimeofday(&t_start, NULL);
    simulate();
    gettimeofday(&t_end, NULL);

    printf("Total elapsed time: %lf seconds\n", WALLTIME(t_end) - WALLTIME(t_start));

    occupancy();

    // Clean up and shut down
    domain_finalize();
    exit(EXIT_SUCCESS);
}
