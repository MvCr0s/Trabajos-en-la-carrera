#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ITER 100

int main(int argc, char *argv[]) {
    int rank, nprocs, ierr;
    double value = 1.0;
    double *recv_values = NULL; // Arreglo para recibir datos
    double total = 0.0;
    int tag = 1000;
    MPI_Status stat;

    // Inicializar MPI
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "Error al inicializar MPI\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Obtener el rango y el número de procesos
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "Error al obtener el rango del proceso\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }

    ierr = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "Error al obtener el tamaño del comunicador\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }

    // El proceso 0 crea el array
    if (rank == 0) {
        recv_values = (double*)malloc(nprocs * sizeof(double));
        if (recv_values == NULL) {
            fprintf(stderr, "Error al asignar memoria para recv_values\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();

    // Multiples fase de cálculo
    for (int i = 0; i < ITER; i++) {
        // Cálculo
        value = sin(value);

        // Operación Colectiva: Gather
        ierr = MPI_Gather(&value, 1, MPI_DOUBLE, recv_values, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (ierr != MPI_SUCCESS) {
            fprintf(stderr, "Error en MPI_Gather\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            free(recv_values);
            return 1;
        }

        // El proceso 0 suma los resultados
        if (rank == 0) {
            total = 0.0;
            for (int j = 0; j < nprocs; j++) {
                total += recv_values[j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;

    // El 0 escribe los resultados de valor y tiempo
    if (rank == 0) {
        printf("Total: %lf\n", total);
        printf("Time: %lf\n", time);

        // Liberar memoria
        free(recv_values);
    }

    ierr = MPI_Finalize();
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "Error al finalizar MPI\n");
        return 1;
    }
    return 0;
}