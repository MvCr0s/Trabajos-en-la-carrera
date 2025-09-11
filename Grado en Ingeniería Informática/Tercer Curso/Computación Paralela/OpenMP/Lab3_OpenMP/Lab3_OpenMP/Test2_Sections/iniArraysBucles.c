#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define SIZE	100000000
#define	SEED	387454

int main() {

	double *A = (double *)malloc( SIZE * sizeof(double) );
	double *B = (double *)malloc( SIZE * sizeof(double) );
	double *C = (double *)malloc( SIZE * sizeof(double) );
	int i;

	double timer_global = omp_get_wtime();
	double timer_init = omp_get_wtime();

	/* 1. Paralelizar la inicialización con bucle paralelo */
#pragma omp for
	for ( i=0; i<SIZE; i++ ) {
		A[i] = i * 3 % 65536 + 33;
		B[i] = (i + SIZE) / 2 % 457 - 17;
		C[i] = 0;
	}

	timer_init = omp_get_wtime() - timer_init;

	/* Suma vectores */
	for (i=0; i<SIZE; i++)
		C[i] = A[i] + B[i];

	timer_global = omp_get_wtime() - timer_global;

	/* Comprobar y escribir timers */
	printf("Primera y última posición: %lf, %lf\n", C[0], C[SIZE-1]);
	printf("Tiempo total: %lf\n", timer_global );
	printf("Tiempo init:  %lf\n", timer_init );

	return 0;
}
