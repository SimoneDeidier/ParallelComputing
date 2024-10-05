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
#define U_prv(i,j) buffers[0][((i)+1)*(N/dims[1]+2)+(j)+1]
#define U(i,j)     buffers[1][((i)+1)*(N/dims[1]+2)+(j)+1]
#define U_nxt(i,j) buffers[2][((i)+1)*(N/dims[1]+2)+(j)+1]
// END: T1b

// Simulation parameters: size, step count, and how often to save the state
int_t M = 256,  // rows
    N = 256,    // cols
    max_iteration = 4000, snapshot_freq = 20;

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
    int local_M = M / dims[0];
    int local_N = N / dims[1];

    // Allocate memory for the local buffers
    buffers[0] = malloc((local_M+2)*(local_N+2)*sizeof(real_t));
    buffers[1] = malloc((local_M+2)*(local_N+2)*sizeof(real_t));
    buffers[2] = malloc((local_M+2)*(local_N+2)*sizeof(real_t));

    // Get the coordinates of the current process in the Cartesian grid
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // Initialize the local buffers with the Gaussian pulse
    int global_i = 0, global_j = 0;
    for (int_t i = 0; i < local_M; i++) {
        for (int_t j = 0; j < local_N; j++) {
            // Calculate global indices
            global_i = coords[0] * local_M + i;
            global_j = coords[1] * local_N + j;
            // Calculate delta (radial distance) adjusted for N x M grid
            real_t delta = sqrt(((global_i - M/2.0) * (global_i - M/2.0)) / (real_t) M + ((global_j - N/2.0) * (global_j - N/2.0)) / (real_t) N);
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
    int local_M = M / dims[0];
    int local_N = N / dims[1];

    for(int_t i = 0; i < local_M; i++) {
        for(int_t j = 0; j < local_N; j++) {
            U_nxt(i, j) = -U_prv(i, j) + 2.0 * U(i, j) + (dt * dt * c * c) / (dx * dy) * (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1) - 4.0 * U(i, j));
        }
    }
// END: T5
}

// TASK: T6
// Communicate the border between processes.
void border_exchange(void) {
// BEGIN: T6
    // Calculate the local dimensions for each process
    int local_M = M / dims[0];
    int local_N = N / dims[1];

    // Define MPI data types for exchanging columns and rows
    MPI_Datatype dom_column, dom_row;
    int north, south, east, west;

    // Create a vector data type for a column
    MPI_Type_vector(local_M, 1, local_N + 2, MPI_DOUBLE, &dom_column);
    MPI_Type_commit(&dom_column);

    // Create a contiguous data type for a row
    MPI_Type_contiguous(local_N, MPI_DOUBLE, &dom_row);
    MPI_Type_commit(&dom_row);

    // Determine the ranks of the neighboring processes
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    // Exchange columns with the west neighbor
    if(west >= 0) {
        MPI_Sendrecv(&U(0, 0), 1, dom_column, west, 0, &U(0, -1), 1, dom_column, west, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    // Exchange columns with the east neighbor
    if(east >= 0) {
        MPI_Sendrecv(&U(0, local_N - 1), 1, dom_column, east, 0, &U(0, local_N), 1, dom_column, east, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    // Exchange rows with the north neighbor
    if(north >= 0) {
        MPI_Sendrecv(&U(0, 0), 1, dom_row, north, 0, &U(-1, 0), 1, dom_row, north, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    // Exchange rows with the south neighbor
    if(south >= 0) {
        MPI_Sendrecv(&U(local_M - 1, 0), 1, dom_row, south, 0, &U(local_M, 0), 1, dom_row, south, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    // Free the MPI data types
    MPI_Type_free(&dom_column);
    MPI_Type_free(&dom_row);
// END: T6
}

// TASK: T7
// Neumann (reflective) boundary condition
void boundary_condition(void) {
// BEGIN: T7
    // Calculate the local dimensions for each process
    int local_M = M / dims[0];
    int local_N = N / dims[1];
    int north, south, west, east;

    // Determine the ranks of the neighboring processes
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    // Set the proper boundary conditions if the process does not have a neighbor in that direction
    if(west < 0) {
        // Reflective boundary condition on the west side
        for(int i = 0; i < local_M; i++) {
            U(i,-1) = U(i,1);
        }
    }
    if(east < 0) {
        // Reflective boundary condition on the east side
        for(int i = 0; i < local_M; i++) {
            U(i,local_N)  = U(i,local_N-2);
        }
    }
    if(north < 0) {
        // Reflective boundary condition on the north side
        for(int j = 0; j < local_N; j++) {
            U(-1,j) = U(1,j);
        }
    }
    if(south < 0) {
        // Reflective boundary condition on the south side
        for(int j = 0; j < local_N; j++) {
            U(local_M,j)  = U(local_M-2,j);
        }
    }
// END: T7
}

// TASK: T8
// Save the present time step in a numbered file under 'data/' using MPI I/O
void domain_save(int_t step) {
// BEGIN: T8
    MPI_File fh;
    MPI_Status status;
    MPI_Offset base_offset;
    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);

    // Calculate the local dimensions for each process
    int local_M = M / dims[0];
    int local_N = N / dims[1];

    // Get the coordinates of the current process in the Cartesian grid
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // Calculate the offset for each process
    base_offset = ((coords[0] * local_M * N) + (coords[1] * local_N)) * sizeof(real_t);

    // Open the file in write mode
    MPI_File_open(cart_comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    // Write the local subgrid to the file using MPI_File_write_at_all
    MPI_Offset offset;
    for (int_t i = 0; i < local_N; i++) {
        offset = base_offset + i * N * sizeof(real_t);
        MPI_File_write_at_all(fh, offset, &U(i, 0), local_N, MPI_DOUBLE, &status);
    }

    // Close the file
    MPI_File_close(&fh);
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
