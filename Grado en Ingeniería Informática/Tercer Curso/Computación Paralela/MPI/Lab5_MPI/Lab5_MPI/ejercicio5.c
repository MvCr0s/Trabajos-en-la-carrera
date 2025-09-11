#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // 1. Leer argumento
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <size_array>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    int size = atoi(argv[1]);

    // 2. Obtener datos del proceso
    int num_procs, my_rank;
    char maquina[MPI_MAX_PROCESSOR_NAME];
    int maquina_len;
    MPI_Get_processor_name(maquina, &maquina_len);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    printf("Proceso: %d de %d, en maquina: %s\n", my_rank, num_procs, maquina);

    // 2. Crear array distribuido (cada proceso un trozo)
    // 2.1. Calcular cuanto espacio necesita cada proceso
    int base_size = size / num_procs;
    int remainder = size % num_procs;
    int my_size;
    if(my_rank<remainder){
	    my_size=base_size+1;
    }else{
	    my_size=base_size;
    }

    // 2.2. Calcular donde empieza cada proceso con respecto al hipotetico array global
    //
    int my_begin;
     if(my_rank<remainder){
            my_begin=my_rank*base_size;
    }else{
            my_begin=(my_size +1 )*remainder+ my_size*(my_rank-remainder);
    }

    int *array_d = (int *)malloc(sizeof(int) * my_size);
    if (array_d == NULL) {
        fprintf(stderr, "Error: Reservando memoria para array distribuido\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // ATENCION:
    // Mis indices locales siempre empiezan en 0, aunque representen otra parte del array global

    // 3. Inicializar mi parte
    for (int i = 0; i < my_size; i++) {
        array_d[i] = i + my_begin;
    }

    // 4. Escribir mi parte
    for (int i = 0; i < my_size; i++) {
        printf("[%d] Pos: %d = %d\n", my_rank, i, array_d[i]);
    }

    free(array_d);
    MPI_Finalize();
    return 0;
}
