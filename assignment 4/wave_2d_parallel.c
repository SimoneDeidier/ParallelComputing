#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "argument_utils.h"

// TASK: T1a
// Include the MPI headerfile
// BEGIN: T1a
#include <mpi.h>    // MPI header file
// END: T1a


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;


// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t* buffers[3] = {NULL, NULL, NULL};

// TASK: T1b
// Declare variables each MPI process will need
int world_rank, world_size; // Rank and size of the MPI communicator
int dims[2]  = {0, 0};      // Dimensions of the Cartesian grid
MPI_Comm cart_comm;         // Cartesian communicator
// BEGIN: T1b
#define U_prv(i,j) buffers[0][((i)+1)*(N+2)+(j)+1]
#define U(i,j)     buffers[1][((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) buffers[2][((i)+1)*(N+2)+(j)+1]
// END: T1b

// Simulation parameters: size, step count, and how often to save the state
int_t M = 256,  // rows
    N = 256,    // cols
    max_iteration = 4000,
    snapshot_freq = 20;

// Wave equation parameters, time step is derived from the space step
const real_t c  = 1.0, dx = 1.0, dy = 1.0;
real_t dt;




// Rotate the time step buffers.
void move_buffer_window(void) {
    real_t* temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// TASK: T4
// Set up our three buffers, and fill two with an initial perturbation
// and set the time step.
void domain_initialize(void) {
// BEGIN: T4
    // Calculate the local dimensions for each process
    int local_N = N / dims[0];
    int local_M = M / dims[1];

    // Allocate memory for the local buffers
    buffers[0] = malloc((local_N+2)*(local_M+2)*sizeof(real_t));
    buffers[1] = malloc((local_N+2)*(local_M+2)*sizeof(real_t));
    buffers[2] = malloc((local_N+2)*(local_M+2)*sizeof(real_t));

    // Get the coordinates of the current process in the Cartesian grid
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // Initialize the local buffers with the Gaussian pulse
    int global_i = 0, global_j = 0;
    for (int_t i = 0; i < local_N; i++) {
        for (int_t j = 0; j < local_M; j++) {
            // Calculate global indices
            global_i = coords[0] * local_N + i;
            global_j = coords[1] * local_M + j;
            // Calculate delta (radial distance) adjusted for N x M grid
            real_t delta = sqrt(((global_i - N/2.0) * (global_i - N/2.0)) / (real_t) N + ((global_j - M/2.0) * (global_j - M/2.0)) / (real_t) M);
            U_prv(i, j) = U(i, j) = exp(-4.0 * delta * delta);
        }
    }

    // Set the time step for 2D case using the CFL condition
    dt = dx * dy / (c * sqrt(dx * dx + dy * dy));
// END: T4
}


// Get rid of all the memory allocations
void domain_finalize(void) {
    free(buffers[0]);
    free(buffers[1]);
    free(buffers[2]);
}


// TASK: T5
// Integration formula
void time_step(void) {
// BEGIN: T5
    // Calculate the local dimensions for each process
    int local_N = N / dims[0];
    int local_M = M / dims[1];

    for(int_t i = 0; i < local_N; i++) {
        for(int_t j = 0; j < local_M; j++) {
            U_nxt(i, j) = -U_prv(i, j) + 2.0 * U(i, j) + (dt * dt * c * c) / (dx * dy) * (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1) - 4.0 * U(i, j));
        }
    }
// END: T5
}

// TASK: T6
// Communicate the border between processes.
void border_exchange(void) {
// BEGIN: T6
    int local_N = N / dims[0];
    int local_M = M / dims[1];

    MPI_Request requests[8];
    int req_count = 0;

    // Get the coordinates of the current process in the Cartesian grid
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // Send to the left, receive from the right
    if (coords[1] > 0) {
        MPI_Isend(&U(0, 0), local_N, MPI_DOUBLE, world_rank - 1, 0, cart_comm, &requests[req_count++]);
        MPI_Irecv(&U(0, -1), local_N, MPI_DOUBLE, world_rank - 1, 0, cart_comm, &requests[req_count++]);
    }
    // Send to the right, receive from the left
    if (coords[1] < dims[1] - 1) {
        MPI_Isend(&U(0, local_M - 1), local_N, MPI_DOUBLE, world_rank + 1, 0, cart_comm, &requests[req_count++]);
        MPI_Irecv(&U(0, local_M), local_N, MPI_DOUBLE, world_rank + 1, 0, cart_comm, &requests[req_count++]);
    }
    // Send to the top, receive from the bottom
    if (coords[0] > 0) {
        MPI_Isend(&U(0, 0), local_M, MPI_DOUBLE, world_rank - dims[1], 0, cart_comm, &requests[req_count++]);
        MPI_Irecv(&U(-1, 0), local_M, MPI_DOUBLE, world_rank - dims[1], 0, cart_comm, &requests[req_count++]);
    }
    // Send to the bottom, receive from the top
    if (coords[0] < dims[0] - 1) {
        MPI_Isend(&U(local_N - 1, 0), local_M, MPI_DOUBLE, world_rank + dims[1], 0, cart_comm, &requests[req_count++]);
        MPI_Irecv(&U(local_N, 0), local_M, MPI_DOUBLE, world_rank + dims[1], 0, cart_comm, &requests[req_count++]);
    }

    // Wait for all non-blocking communications to complete
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
// END: T6
}


// TASK: T7
// Neumann (reflective) boundary condition
void boundary_condition(void) {
// BEGIN: T7
    int local_N = N / dims[0];
    int local_M = M / dims[1];

    // Get the coordinates of the current process in the Cartesian grid
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // Apply boundary conditions only on the processes at the boundaries
    if (coords[1] == 0) { // Left boundary
        for (int_t i = 0; i < local_N; i++) {
            U(i, -1) = U(i, 1);
        }
    }
    if (coords[1] == dims[1] - 1) { // Right boundary
        for (int_t i = 0; i < local_N; i++) {
            U(i, local_M) = U(i, local_M - 2);
        }
    }
    if (coords[0] == 0) { // Top boundary
        for (int_t j = 0; j < local_M; j++) {
            U(-1, j) = U(1, j);
        }
    }
    if (coords[0] == dims[0] - 1) { // Bottom boundary
        for (int_t j = 0; j < local_M; j++) {
            U(local_N, j) = U(local_N - 2, j);
        }
    }
// END: T7
}


// TASK: T8
// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
// BEGIN: T8
    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);

    MPI_File fh;
    MPI_Offset offset;
    MPI_Datatype filetype;

    // Calculate the local dimensions for each process
    int local_N = N / dims[0];
    int local_M = M / dims[1];

    // Get the coordinates of the current process in the Cartesian grid
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // Create the file and set the view
    MPI_File_open(cart_comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    // Define the file type for the subarray
    int gsizes[2] = {N, M};
    int lsizes[2] = {local_N, local_M};
    int starts[2] = {coords[0] * local_N, coords[1] * local_M};

    MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    // Set the file view
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    // Write the local array to the file
    for (int_t i = 0; i < local_N; i++) {
        MPI_File_write(fh, &U(i, 0), local_M, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }

    // Close the file
    MPI_File_close(&fh);

    MPI_Type_free(&filetype);
// END: T8
}


// Main time integration.
void simulate(void) {
    // Go through each time step
    for(int_t iteration=0; iteration <= max_iteration; iteration++) {
        if((iteration % snapshot_freq) == 0) {
            domain_save(iteration / snapshot_freq);
        }

        // Derive step t+1 from steps t and t-1
        border_exchange();
        boundary_condition();
        time_step();

        // Rotate the time step buffers
        move_buffer_window();
    }
}


int main (int argc, char **argv) {

    // TASK: T1c
    // Initialise MPI
    // BEGIN: T1c
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // END: T1c


    // TASK: T3
    // Distribute the user arguments to all the processes
    // BEGIN: T3
    OPTIONS *options = NULL;
    if (world_rank == 0) {
        // Parse the command line arguments on the root process
        options = parse_args(argc, argv);
        if (!options) {
            // If argument parsing fails, abort the MPI program
            fprintf(stderr, "Argument parsing failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        // Set the simulation parameters from the parsed arguments
        M = options->M;
        N = options->N;
        max_iteration = options->max_iteration;
        snapshot_freq = options->snapshot_frequency;
    }

    // Broadcast the parsed arguments to all processes
    MPI_Bcast(&M, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_freq, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

    // Set up Cartesian communication topology
    MPI_Dims_create(world_size, 2, dims); // Create a 2D grid of processes
    int periods[2] = {0, 0}; // No wrap-around in any dimension
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm); // Create the Cartesian communicator
    // END: T3

    // Set up the initial state of the domain
    domain_initialize();


    struct timeval t_start, t_end;

    // TASK: T2
    // Time your code
    // BEGIN: T2
    gettimeofday(&t_start, NULL);   // Start the timer   
    simulate(); // Run the simulation
    gettimeofday(&t_end, NULL); // Stop the timer
    double local_time = WALLTIME(t_end) - WALLTIME(t_start);    // Calculate the local elapsed time
    double total_time = 0.0;    // Variable to store the total time across all processes
    // Reduce the local times to get the total time across all processes
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // If this is the root process, calculate and print the mean simulation time
    if(world_rank == 0) {
        double mean_time = total_time / world_size;
        printf("Mean simulation time: %f seconds\n", mean_time);
    }
    // END: T2

    // Clean up and shut down
    domain_finalize();

    // TASK: T1d
    // Finalise MPI
    // BEGIN: T1d
    MPI_Finalize();
    // END: T1d

    exit(EXIT_SUCCESS);

}
