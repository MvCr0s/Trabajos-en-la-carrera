#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "rng.c"

#define SIZE 200000000
#define MODULO 65535

int main() {
    rng_t rng = rng_new(823531);  
    rng_t rng2 = rng;

    int *array = (int *)malloc(SIZE * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Error al asignar memoria\n");
        return 1;
    }

    int sum = 0;  // Variable de suma truncada
    int i;

    for (i = 0; i < SIZE; i++) {
        array[i] = (int)(rng_next(&rng) * 1000);  // [0, 999]
    }

    for (i = 0; i < SIZE; i++) {
        sum = (sum + array[i]) % MODULO;  
    }
    
    printf("Suma truncada (mod %d): %u\n", MODULO, sum);

    printf("El siguiente numero aleatorio es: %f\n", rng_next(&rng));

    rng_skip(&rng2, SIZE); // Salta SIZE numeros aleatorios
    printf("El siguiente numero aleatorio en la segunda serie: %f\n", rng_next(&rng2));

    free(array);
    return 0;
}
