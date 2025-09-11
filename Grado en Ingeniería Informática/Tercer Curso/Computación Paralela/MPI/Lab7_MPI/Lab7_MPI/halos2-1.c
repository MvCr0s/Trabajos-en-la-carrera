#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>    // Include MPI header
#include <string.h> // For memcpy

// Assuming rng.c is available and provides these functions:
// typedef struct { ... } rng_t;
// rng_t rng_new(unsigned long long seed);
// double rng_next_between(rng_t *rng, double min, double max);
#include "rng.c"

// Renamed helper function to print a vector (now used by rank 0 for global vector)
void imprimirVector(const char *label, double v[], int n) {
    printf("%s:\n", label);
    for (int i = 0; i < n; i++) {
        printf("%.2lf ", v[i]);
    }
    printf("\n");
}

// Helper function for debugging: print local vector with ghost cells
// void imprimirVectorLocal(const char* label, double v_local[], int local_n_with_ghosts, int rank, int iter) {
//      printf("[Rank %d, Iter %d] %s (size %d): ", rank, iter, label, local_n_with_ghosts);
//      for (int i = 0; i < local_n_with_ghosts; i++) {
//          if (i == 0 || i == local_n_with_ghosts - 1) {
//              printf("(%.2lf) ", v_local[i]); // Mark ghost cells
//          } else {
//              printf("%.2lf ", v_local[i]);
//          }
//      }
//      printf("\n");
//      fflush(stdout); // Ensure output is flushed immediately
// }


int main(int argc, char *argv[]) {

    int world_rank, world_size;

    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int global_size = 0;
    int num_iters = 0;
    double *v_global = NULL;       // Only rank 0 will allocate the full initial vector
    double *v_global_temp = NULL; // Rank 0 needs a buffer to gather results for printing

    // --- Argument Parsing and Broadcasting ---
    if (world_rank == 0) {
        if (argc < 3) {
            fprintf(stderr, "Uso: mpirun -np <N> %s <tamaño del vector> <iteraciones>\n", argv[0]);
            global_size = -1; // Use a flag value
            // Signal other processes to exit before calling MPI_Abort (best effort)
            int params[2] = {global_size, num_iters};
            MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        global_size = atoi(argv[1]);
        num_iters = atoi(argv[2]);

        if (global_size <= 0 || num_iters < 0 || global_size < world_size) {
             if (global_size <= 0 || num_iters < 0)
                fprintf(stderr, "Tamaño del vector e iteraciones deben ser positivos.\n");
             if (global_size < world_size)
                 fprintf(stderr, "Tamaño del vector (%d) debe ser al menos el número de procesos (%d).\n", global_size, world_size);
             global_size = -1; // Flag for abort
        }
         // Broadcast size and iterations to all processes
        int params[2] = {global_size, num_iters};
        MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);
        global_size = params[0];
        num_iters = params[1];

        if (global_size == -1) { // Check if root decided to abort
             MPI_Finalize();
             return EXIT_FAILURE;
        }

        // --- Root Process: Initialize Global Vector ---
        v_global = (double *)malloc(global_size * sizeof(double));
        v_global_temp = (double *)malloc(global_size * sizeof(double)); // Buffer for Gatherv
        if (!v_global || !v_global_temp) {
            perror("Failed to allocate global vector(s)");
             MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // Abort all processes
        }
        rng_t rng = rng_new(0); // Seed RNG - IMPORTANT: Seed must be the same for reproducibility
        for (int i = 0; i < global_size; i++) {
            v_global[i] = rng_next_between(&rng, 0, 100);
        }
        // Use the renamed print function
        imprimirVector("Vector inicial", v_global, global_size);

    } else {
        // --- Non-Root Processes: Receive Broadcasted Params ---
        int params[2];
        MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);
        global_size = params[0];
        num_iters = params[1];

        if (global_size == -1) { // Check if root signaled abort
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        // Non-root processes don't need the gather buffer
        v_global = NULL;
        v_global_temp = NULL;
    }

     // --- Determine Local Size and Offsets for Data Distribution ---
    // ALL processes need to calculate this for Scatterv and Gatherv logic
    int base_chunk_size = global_size / world_size;
    int remainder = global_size % world_size;
    int local_size = base_chunk_size + (world_rank < remainder ? 1 : 0);
    int start_index = world_rank * base_chunk_size + (world_rank < remainder ? world_rank : remainder);

    // Allocate memory for local vector chunks + 2 ghost cells
    int local_size_with_ghosts = local_size + 2;
    double *v_local = (double *)malloc(local_size_with_ghosts * sizeof(double));
    double *temp_local = (double *)malloc(local_size_with_ghosts * sizeof(double));
    if (!v_local || !temp_local) {
        perror("Failed to allocate local vector");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // --- Calculate counts and displacements for Scatterv/Gatherv ---
    // Needed by ALL processes for Gatherv call, even if args only used by root
    int *counts = (int *)malloc(world_size * sizeof(int));
    int *displs = (int *)malloc(world_size * sizeof(int));
    if (!counts || !displs) {
         perror("Failed to allocate counts/displs arrays");
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int current_displ = 0;
    for (int i = 0; i < world_size; ++i) {
        counts[i] = base_chunk_size + (i < remainder ? 1 : 0);
        displs[i] = current_displ;
        current_displ += counts[i];
    }

    // --- Scatter Data using MPI_Scatterv ---
    MPI_Scatterv(v_global,           // Send buffer (only valid on root)
                 counts,             // Array of send counts
                 displs,             // Array of displacements
                 MPI_DOUBLE,         // Data type
                 &v_local[1],        // Receive buffer (start at index 1)
                 local_size,         // Receive count for this process
                 MPI_DOUBLE,         // Data type
                 0,                  // Root process rank
                 MPI_COMM_WORLD);

    // Rank 0 no longer needs the initial global vector (will use v_global_temp)
    if (world_rank == 0) {
        free(v_global);
        v_global = NULL;
    }

    // --- Determine Neighbors ---
    int left_neighbor = (world_rank == 0) ? MPI_PROC_NULL : world_rank - 1;
    int right_neighbor = (world_rank == world_size - 1) ? MPI_PROC_NULL : world_rank + 1;

    // --- Main Iteration Loop ---
    MPI_Status status_left, status_right; // Status objects for Sendrecv

    for (int iter = 0; iter < num_iters; iter++) {

        // 1. --- Ghost Cell Exchange ---
         MPI_Sendrecv(&v_local[1],             // Send buffer (first real element)
                     1,                        // Send count
                     MPI_DOUBLE,               // Send type
                     left_neighbor,            // Destination rank (left)
                     0,                        // Send tag (iter 0)
                     &v_local[local_size + 1], // Receive buffer (right ghost cell)
                     1,                        // Receive count
                     MPI_DOUBLE,               // Receive type
                     right_neighbor,           // Source rank (right)
                     0,                        // Receive tag (iter 0) - Must match send tag
                     MPI_COMM_WORLD,
                     &status_right);           // Status object

        MPI_Sendrecv(&v_local[local_size],    // Send buffer (last real element)
                     1,                       // Send count
                     MPI_DOUBLE,              // Send type
                     right_neighbor,          // Destination rank (right)
                     1,                       // Send tag (iter 1)
                     &v_local[0],             // Receive buffer (left ghost cell)
                     1,                       // Receive count
                     MPI_DOUBLE,              // Receive type
                     left_neighbor,           // Source rank (left)
                     1,                       // Receive tag (iter 1) - Must match send tag
                     MPI_COMM_WORLD,
                     &status_left);           // Status object


        // 2. --- Local Computation ---
        for (int i = 1; i <= local_size; i++) {
            int global_index = start_index + i - 1; // Calculate global index

            if (global_index == 0) {
                temp_local[i] = (v_local[i] + v_local[i + 1]) / 2.0;
            } else if (global_index == global_size - 1) {
                temp_local[i] = (v_local[i - 1] + v_local[i]) / 2.0;
            } else {
                // Indices i-1, i, i+1 are valid due to ghost cells
                temp_local[i] = (v_local[i - 1] + v_local[i] + v_local[i + 1]) / 3.0;
            }
        }

        // 3. --- Copy Back ---
        // Use memcpy for potential efficiency
        memcpy(&v_local[1], &temp_local[1], local_size * sizeof(double));
        // Equivalent loop:
        // for (int i = 1; i <= local_size; i++) {
        //      v_local[i] = temp_local[i];
        // }

        // 4. --- Gather results to Rank 0 for printing ---
        MPI_Gatherv(&v_local[1],       // Send buffer (start of local data)
                   local_size,        // Send count for this process
                    MPI_DOUBLE,        // Send type
                    v_global_temp,     // Receive buffer (only significant on root)
                    counts,            // Array of receive counts (how many from each rank)
                    displs,            // Array of displacements (where to put data in recv buf)
                    MPI_DOUBLE,        // Receive type
                    0,                 // Root process rank
                    MPI_COMM_WORLD);

        // 5. --- Rank 0 Prints the gathered vector ---
        if (world_rank == 0) {
            char label[50];
            snprintf(label, sizeof(label), "Iteración %d", iter + 1);
            imprimirVector(label, v_global_temp, global_size);
        }

         // Optional Barrier: If output seems jumbled between iterations/ranks
         // MPI_Barrier(MPI_COMM_WORLD);

    } // End of iteration loop

    // --- Calculate Local Sum ---
    double local_sum = 0.0;
    for (int i = 1; i <= local_size; i++) {
        local_sum += v_local[i]; // Sum the final state from v_local
    }

    // --- Reduce Local Sums to Global Sum on Root ---
    double global_sum = 0.0;
    MPI_Reduce(&local_sum,        // Send buffer (local sum)
               &global_sum,       // Receive buffer (global sum, only valid on root)
               1,                 // Count
               MPI_DOUBLE,        // Data type
               MPI_SUM,           // Reduction operation
               0,                 // Root process rank
               MPI_COMM_WORLD);

    // --- Print Final Result (only on root) ---
    if (world_rank == 0) {
        printf("Suma de los valores del vector resultante: %.2lf\n", global_sum);
    }

    // --- Cleanup ---
    free(v_local);
    free(temp_local);
    free(counts); // Free counts/displs on all processes
    free(displs);
    if (world_rank == 0) {
        free(v_global_temp); // Free the gather buffer on root
    }

    MPI_Finalize();

    return 0;
}