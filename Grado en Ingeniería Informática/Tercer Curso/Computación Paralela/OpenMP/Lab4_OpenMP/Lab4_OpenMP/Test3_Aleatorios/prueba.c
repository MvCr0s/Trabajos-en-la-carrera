#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "rng.c"

#define ROWS 1000
#define COLS 1000
#define MODULO 65535

int main() {
    rng_t rng_main = rng_new(12345);  // RNG principal
    int *matrix = (int *) malloc(ROWS * COLS * sizeof(int));
    
    if (matrix == NULL) {
        fprintf(stderr, "Error al asignar memoria\n");
        return 1;
    }

    long sum = 0; // Variable de suma truncada

    // 🔹 PARALELIZACIÓN DE LA GENERACIÓN DE NÚMEROS ALEATORIOS 🔹
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // Cada hilo obtiene su propio RNG basado en el principal
        rng_t rng = rng_main;
        rng_skip(&rng, thread_id * (ROWS * COLS) / num_threads); 

        // Cada hilo genera sus números en su parte de la matriz
        #pragma omp for
        for (int i = 0; i < ROWS * COLS; i++) {
            matrix[i] = (long) (rng_next(&rng) * 1000);  // [0, 999]
        }
    }

    // 🔹 PARALELIZACIÓN DEL CÁLCULO DE `CHECKSUM` 🔹
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < ROWS * COLS; i++) {
        sum = (sum + matrix[i]) % MODULO;
    }

    printf("Suma truncada (mod %d): %ld\n", MODULO, sum);
    printf("El siguiente número aleatorio es: %f\n", rng_next(&rng_main));

    // 🔹 Generación determinista para la segunda serie 🔹
    rng_t rng2 = rng_main;
    rng_skip(&rng2, ROWS * COLS);  // Salta ROWS * COLS números aleatorios
    printf("El siguiente número aleatorio en la segunda serie: %f\n", rng_next(&rng2));

    free(matrix);
    return 0;
}

