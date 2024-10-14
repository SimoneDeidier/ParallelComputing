#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <sys/time.h>

// TASK: T1a
// Include the pthreads library
// BEGIN: T1a
#include <pthread.h>
// END: T1a

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;


// TASK: T1b
// Pthread management
// BEGIN: T1b
int_t n_threads = 1;
pthread_barrier_t barrier;
// END: T1b

// Performance measurement
struct timeval t_start, t_end;
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec) 

// Simulation parameters: size, step count, and how often to save the state
const int_t N = 1024, max_iteration = 4000, snapshot_freq = 20;

// Wave equation parameters, time step is derived from the space step
const real_t c  = 1.0, h  = 1.0;
real_t dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t* buffers[3] = {NULL, NULL, NULL};

#define U_prv(i,j) buffers[0][((i)+1)*(N+2)+(j)+1]
#define U(i,j)     buffers[1][((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) buffers[2][((i)+1)*(N+2)+(j)+1]


// Rotate the time step buffers.
void move_buffer_window(void) {
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize(void) {
    buffers[0] = malloc((N+2)*(N+2)*sizeof(real_t));
    buffers[1] = malloc((N+2)*(N+2)*sizeof(real_t));
    buffers[2] = malloc((N+2)*(N+2)*sizeof(real_t));

    for(int_t i = 0; i < N; i++) {
        for(int_t j = 0; j < N; j++) {
            real_t delta = sqrt(((i - N / 2) * (i - N / 2) + (j - N / 2) * (j - N / 2)) / (real_t)N );
            U_prv(i,j) = U(i,j) = exp(-4.0 * delta * delta);
        }
    }

    // Set the time step
    dt = (h * h) / (4.0 * c * c);
}


// Get rid of all the memory allocations
void domain_finalize(void) {
    free(buffers[0]);
    free(buffers[1]);
    free(buffers[2]);
}


// TASK: T3
// Integration formula
void time_step(int_t thread_id) {
// BEGIN: T3
    // Calculate the start and end indices for the current thread
    int_t start = (N / n_threads) * thread_id;
    int_t end = (thread_id == n_threads - 1) ? N : start + (N / n_threads);

    // Loop over the assigned rows for the current thread
    for(int_t i = start; i < end; i++) {
        // Loop over all columns
        for(int_t j = 0; j < N; j++) {
            // Compute the next time step using the wave equation
            U_nxt(i,j) = -U_prv(i,j) + 2.0 * U(i,j) + (dt * dt * c * c) / (h * h) * (U(i - 1,j) + U(i + 1,j) + U(i,j - 1) + U(i,j + 1) - 4.0 * U(i,j));
        }
    }

    // Synchronize threads at the barrier
    pthread_barrier_wait(&barrier);
// END: T3
}


// TASK: T4
// Neumann (reflective) boundary condition
void boundary_condition(int_t thread_id) {
// BEGIN: T4
    // Calculate the start and end indices for the current thread
    int_t start = (N / n_threads) * thread_id;
    int_t end = (thread_id == n_threads - 1) ? N : start + (N / n_threads);

    // Apply boundary conditions for the left and right edges
    for (int_t i = start; i < end; i++) {
        U(i, -1) = U(i, 1);      // Reflective boundary on the left edge
        U(i, N) = U(i, N - 2);   // Reflective boundary on the right edge
    }

    // Apply boundary conditions for the top edge if this is the first thread
    if (thread_id == 0) {
        for (int_t j = 0; j < N; j++) {
            U(-1, j) = U(1, j);  // Reflective boundary on the top edge
        }
    }

    // Apply boundary conditions for the bottom edge if this is the last thread
    if (thread_id == n_threads - 1) {
        for (int_t j = 0; j < N; j++) {
            U(N, j) = U(N - 2, j);  // Reflective boundary on the bottom edge
        }
    }

    // Synchronize threads at the barrier
    pthread_barrier_wait(&barrier);
// END: T4
}


// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    FILE* out = fopen(filename, "wb");
    for(int_t i = 0; i < N; i++) {
        fwrite(&U(i,0), sizeof(real_t), N, out);
    }
    fclose(out);
}


// TASK: T5
// Main loop
void* simulate(void *id) {
// BEGIN: T5
    int_t thread_id = *(int_t *)id;

    // Go through each time step
    for(int_t iteration = 0; iteration <= max_iteration; iteration++) {
        // Ensure only one thread saves the state
        if(thread_id == 0 && (iteration % snapshot_freq) == 0) {
            domain_save(iteration / snapshot_freq);
        }

        // Derive step t+1 from steps t and t-1
        boundary_condition(thread_id);
        time_step(thread_id);

        // Synchronize threads at the barrier
        pthread_barrier_wait(&barrier);

        // Rotate the time step buffers (only one thread needs to do this)
        if (thread_id == 0) {
            move_buffer_window();
        }

        // Synchronize threads at the barrier
        pthread_barrier_wait(&barrier);
    }

    return NULL;
// END: T5
}


// Main time integration loop
int main(int argc, char **argv) {
    // Number of threads is an optional argument, sanity check its value
    if(argc > 1) {
        n_threads = strtol(argv[1], NULL, 10);
        if(errno == EINVAL) {
            fprintf(stderr, "'%s' is not a valid thread count\n", argv[1]);
        }
        if(n_threads < 1) {
            fprintf(stderr, "Number of threads must be >0\n");
            exit(EXIT_FAILURE);
        }
    }

    // TASK: T1c
    // Initialise pthreads
    // BEGIN: T1c
    pthread_barrier_init(&barrier, NULL, n_threads);
    // END: T1c

    // Set up the initial state of the domain
    domain_initialize();

    // Time the execution
    gettimeofday(&t_start, NULL);

    // TASK: T2
    // Run the integration loop
    // BEGIN: T2
    pthread_t threads[n_threads];
    int_t thread_ids[n_threads];

    // Create threads
    for (int_t i = 0; i < n_threads; i++) {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, simulate, (void *)&thread_ids[i]) != 0) {
            fprintf(stderr, "Error creating thread %ld\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Join threads
    for (int_t i = 0; i < n_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Error joining thread %ld\n", i);
            exit(EXIT_FAILURE);
        }
    }
    // END: T2

    // Report how long we spent in the integration stage
    gettimeofday(&t_end , NULL);
    printf("%lf seconds elapsed with %ld threads\n", WALLTIME(t_end) - WALLTIME(t_start), n_threads);

    // Clean up and shut down
    domain_finalize();

    // TASK: T1d
    // Finalise pthreads
    // BEGIN: T1d
    pthread_barrier_destroy(&barrier);
    // END: T1d

    exit(EXIT_SUCCESS);
}
