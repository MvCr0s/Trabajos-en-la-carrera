#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include "rng.c"

void imprimirVector(double v[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%.2lf ", v[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        fprintf(stderr, "Uso: %s <tamaño del vector> <iteraciones>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int size = atoi(argv[1]);
    int num_iters = atoi(argv[2]);

    double v[size], temp[size];
    rng_t rng = rng_new(0);

    for (int i = 0; i < size; i++) {
        v[i] = rng_next_between(&rng, 0, 100);
    }

    printf("Vector inicial:\n");
    imprimirVector(v, size);

    for (int iter = 0; iter < num_iters; iter++) {
        
        for (int i = 0; i < size; i++) {
            if (i==0)
                temp[i] = (v[i] + v[i + 1]) / 2.0; 
            else if (i==size-1)
                temp[i] = (v[i - 1] + v[i]) / 2.0; 
            else
                temp[i] = (v[i - 1] + v[i] + v[i + 1]) / 3.0;
        }

        // Copiar los nuevos valores al vector original
        for (int i = 0; i < size; i++) {
            v[i] = temp[i];
        }

        printf("Iteración %d:\n", iter + 1);
        imprimirVector(v, size);
    }

    double suma = 0.0;
    for (int i = 0; i < size; i++) {
        suma += v[i];
    }
    printf("Suma de los valores del vector resultante: %.2lf\n", suma);

    return 0;
}
