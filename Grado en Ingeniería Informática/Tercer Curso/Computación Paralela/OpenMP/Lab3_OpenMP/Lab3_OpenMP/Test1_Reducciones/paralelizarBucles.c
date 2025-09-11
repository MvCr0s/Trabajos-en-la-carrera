#include<stdio.h>
#include<stdlib.h>
/* 1. Include de biblioteca OpenMP */

#define SIZE	300000000
int main() {

	int *v = (int *)malloc( SIZE * sizeof(int) );
	int i;
	int suma = 0;

#pragma omp parallel for private(i) shared(v)
	/* 2. Paralelizar la inicializaci�n. Medir tiempos con diferente n�mero de threads */
	for ( i=0; i<SIZE; i++ ) v[i] = i;
	/* 3. Paralelizar la suma, utilizando la cl�usula de reducci�n. Medir tiempos 
			con diferentes n�mero de threads.
			Atentos al valor "Resultado final" en cada prueba. Si est� todo bien 
			deber�a ser siempre igual en todas las pruebas.
	*/

#pragma omp parallel for reduction(+:suma)
	for ( i=0; i<SIZE; i++ ) suma = ( suma + v[i] )%65535 ;
	suma = suma %65535;
	printf( "Resultado final: %d\n", suma );

	return 0;

}








