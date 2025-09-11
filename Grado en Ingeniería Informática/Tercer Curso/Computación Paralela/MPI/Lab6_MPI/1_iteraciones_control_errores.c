#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ITER 100

int main(int argc, char *argv[]) {
    int rank, nprocs, ierr;
    double value = 1.0;
    double recv_value;
    double total = 0;
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

    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();

    // Multiples fase de cálculo y comunicación
    int i;
    for (i = 0; i < ITER; i++) {
        // Cálculo
        value = sin(value);

        // Comunicación
        // Todos menos el 0 mandan su valor
        if (rank != 0) {
            ierr = MPI_Send(&value, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
            if (ierr != MPI_SUCCESS) {
                fprintf(stderr, "Error al enviar el mensaje desde el proceso %d\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }
        } else {
            int j;
            // El 0 recibe mensajes de cada uno y acumula los valores
            for (j = 1; j < nprocs; j++) {
                ierr = MPI_Recv(&recv_value, 1, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &stat);
                if (ierr != MPI_SUCCESS) {
                    fprintf(stderr, "Error al recibir el mensaje en el proceso 0 desde el proceso %d\n", j);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    MPI_Finalize();
                    return 1;
                }
                total = total + recv_value;
            }
            // El 0 también suma su propio valor
            total = total + value;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;

    // El 0 escribe los resultados de valor y tiempo
    if (rank == 0) {
        printf("Total: %lf\n", total);
        printf("Time: %lf\n", time);
    }

    ierr = MPI_Finalize();
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "Error al finalizar MPI\n");
        return 1;
    }
    return 0;
}
