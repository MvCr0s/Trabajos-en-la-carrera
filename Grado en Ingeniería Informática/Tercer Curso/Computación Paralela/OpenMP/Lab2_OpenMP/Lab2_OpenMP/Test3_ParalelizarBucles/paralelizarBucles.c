#include<stdio.h>
#include<stdlib.h>
/* 1. Añadir include de biblioteca OpenMP */
#include <omp.h>
#define SIZE	200000000
int main() {

	double total=omp_get_wtime();
	int *v = (int *)malloc( SIZE * sizeof(int) );
	int i;
	int suma = 0;
	

	
	/* Inicialización */
	
	#pragma omp parallel for private(i) shared(v)
	for ( i=0; i<SIZE; i++ ) v[i] = 0;

	/* 2. Añadir directiva de bucle paralelo: Con vector v compartido e índice i privado */
	double parallel=omp_get_wtime();
	#pragma omp parallel for private(i) shared(v)
		
	for ( i=0; i<SIZE; i++ ) v[i] = i;
	
	parallel=omp_get_wtime() - parallel;
	total=omp_get_wtime() - total;
	/* Suma secuencial */
	printf("total: %lf, Paralelo: %lf\n", total, parallel);
	for ( i=0; i<SIZE; i++ ) suma = ( suma + v[i] ) % 65535;
	
	

	
	
	
	printf( "Resultado final: %d\n", suma );

	return 0;



}
