#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

// TASK: T1a
// Include the MPI headerfile
// BEGIN: T1a
#include <mpi.h>        // MPI header file
// END: T1a


// Option to change numerical precision.
typedef int64_t int_t;
typedef double real_t;


// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
int world_rank, world_size;         // Rank and size of the MPI communicator
// END: T1b


// Simulation parameters: size, step count, and how often to save the state.
const int_t N = 65536, max_iteration = 100000, snapshot_freq = 500;

// Wave equation parameters, time step is derived from the space step.
const real_t c  = 1.0, dx = 1.0;
real_t dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary.
real_t* buffers[3] = {NULL, NULL, NULL};


#define U_prv(i) buffers[0][(i)+1]
#define U(i)     buffers[1][(i)+1]
#define U_nxt(i) buffers[2][(i)+1]


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)


// TASK: T8
// Save the present time step in a numbered file under 'data/'.
void domain_save(int_t step) {
// BEGIN: T8
    if (world_rank == 0) {
        char filename[256];
        sprintf(filename, "data/%.5ld.dat", step);
        FILE *out = fopen(filename, "wb");
        fwrite(&U(0), sizeof(real_t), N, out);
        fclose(out);
    }
// END: T8
}


// TASK: T3
// Allocate space for each process' sub-grids
// Set up our three buffers, fill two with an initial cosine wave,
// and set the time step.
void domain_initialize(void){
// BEGIN: T3
    int_t local_N = N / world_size;
    if (world_rank == world_size - 1) {
        local_N += N % world_size; // Last process takes the remaining points
    }
    buffers[0] = malloc((local_N + 2) * sizeof(real_t));
    buffers[1] = malloc((local_N + 2) * sizeof(real_t));
    buffers[2] = malloc((local_N + 2) * sizeof(real_t));

    int_t start = world_rank * local_N;
    for (int_t i = 0; i < local_N; i++) {
        U_prv(i) = U(i) = cos(M_PI * (start + i) / (real_t)N);
    }

    // Root process allocates space for the entire domain
    if (world_rank == 0) {
        buffers[0] = realloc(buffers[0], (N + 2) * sizeof(real_t));
        buffers[1] = realloc(buffers[1], (N + 2) * sizeof(real_t));
        buffers[2] = realloc(buffers[2], (N + 2) * sizeof(real_t));
    }

    // Set the time step for 1D case.
    dt = dx / c;
}


// Return the memory to the OS.
void domain_finalize(void) {
    free(buffers[0]);
    free(buffers[1]);
    free(buffers[2]);
}


// Rotate the time step buffers.
void move_buffer_window(void) {
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// TASK: T4
// Derive step t+1 from steps t and t-1.
void time_step(void){
// BEGIN: T4
    int_t local_N = N / world_size;
    if (world_rank == world_size - 1) {
        local_N += N % world_size; // Last process takes the remaining points
    }
    for(int_t i = 0; i < local_N; i++){
        U_nxt(i) = -U_prv(i) + 2.0*U(i) + (dt*dt*c*c)/(dx*dx) * (U(i-1)+U(i+1)-2.0*U(i));
    }
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition.
void boundary_condition(void) {
// BEGIN: T6
    int_t local_N = N / world_size;
    if (world_rank == world_size - 1) {
        local_N += N % world_size; // Last process takes the remaining points
    }

    // Apply boundary condition only on the processes at the boundaries
    if(world_rank == 0) {
        U(-1) = U(1); // Left boundary
    }
    if(world_rank == world_size - 1) {
        U(local_N) = U(local_N - 2); // Right boundary
    }
// END: T6
}


// TASK: T5
// Communicate the border between processes.
void border_exchange(void) {
// BEGIN: T5
    MPI_Request request[4];
    int_t local_N = N / world_size;
    if (world_rank == world_size - 1) {
        local_N += N % world_size; // Last process takes the remaining points
    }

    // Send left ghost cell to the left neighbor
    if(world_rank > 0) {
        MPI_Isend(&U(0), 1, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&U(-1), 1, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, &request[1]);
    }

    // Send right ghost cell to the right neighbor
    if(world_rank < world_size - 1) {
        MPI_Isend(&U(local_N - 1), 1, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&U(local_N), 1, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, &request[3]);
    }

    // Wait for all communications to complete
    if(world_rank > 0) {
        MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        MPI_Wait(&request[1], MPI_STATUS_IGNORE);
    }
    if(world_rank < world_size - 1) {
        MPI_Wait(&request[2], MPI_STATUS_IGNORE);
        MPI_Wait(&request[3], MPI_STATUS_IGNORE);
    }
// END: T5
}


// TASK: T7
// Every process needs to communicate its results
// to root and assemble it in the root buffer
void send_data_to_root(){
// BEGIN: T7
    // Calculate the number of elements each process will handle
    int_t local_N = N / world_size;
    if (world_rank == world_size - 1) {
        local_N += N % world_size; // Last process takes the remaining points
    }

    // Number of elements to send
    int sendcount = local_N;
    // Pointer to the data to send
    real_t* sendbuf = &U(0);

    // Buffer to receive data on the root process
    real_t* recvbuf = NULL;
    // Array to store the number of elements each process will send
    int recvcounts[world_size];
    // Array to store the displacements at which to place the incoming data
    int displs[world_size];

    // If this is the root process, prepare to receive data from all processes
    if(world_rank == 0) {
        for(int i = 0; i < world_size; i++) {
            recvcounts[i] = N / world_size;
            if (i == world_size - 1) {
                recvcounts[i] += N % world_size; // Last process takes the remaining points
            }
            displs[i] = i * (N / world_size);
        }
        // Allocate memory to receive the entire domain
        recvbuf = malloc(N * sizeof(real_t));
    }

    // Gather data from all processes to the root process
    MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // If this is the root process, copy the received data into the main buffer
    if(world_rank == 0) {
        for(int i = 0; i < N; i++) {
            U(i) = recvbuf[i];
        }
        // Free the receive buffer
        free(recvbuf);
    }
// END: T7
}


// Main time integration.
void simulate(void){
    // Go through each time step.
    for(int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        if((iteration % snapshot_freq) == 0) {
            send_data_to_root();
            domain_save(iteration / snapshot_freq);
        }

        // Derive step t+1 from steps t and t-1.
        border_exchange();
        boundary_condition();
        time_step();
        move_buffer_window();
    }
}


int main (int argc, char **argv) {
// TASK: T1c
// Initialise MPI
// BEGIN: T1c
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
// END: T1c
    
    struct timeval t_start, t_end;

    domain_initialize();

// TASK: T2
// Time your code
// BEGIN: T2
    // synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    // start the timer
    gettimeofday(&t_start, NULL);
    simulate();
    // end the timer
    gettimeofday(&t_end, NULL);
    // calculate elapsed time
    double time = WALLTIME(t_end) - WALLTIME(t_start);
    double max_time;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(world_rank == 0) {
        printf("Elapsed time: %f seconds\n", max_time);
    }
// END: T2
   
    domain_finalize();

// TASK: T1d
// Finalise MPI
// BEGIN: T1d
    MPI_Finalize();
// END: T1d

    exit(EXIT_SUCCESS);

}
