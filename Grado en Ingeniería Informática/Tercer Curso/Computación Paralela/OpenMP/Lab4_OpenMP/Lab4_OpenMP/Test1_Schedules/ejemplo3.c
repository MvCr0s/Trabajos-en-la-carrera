
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#define SIZE	1024
#define SEED	6834723

int main() {
	int i,j,k;

	double V[ SIZE ][ SIZE ];

	srand48( SEED );

	for (i=0; i<SIZE; i++)
		for (j=0; j<SIZE; j++) {
			V[i][j] = drand48();
		}

	/* Paralelizar esta parte del código */
	//#pragma omp for nowait
	//#pragma omp parallel for ordered schedule( static )
	#pragma omp parallel for collapse(2)
	for (i=0; i<SIZE; i++)
		for (j=0; j<SIZE; j++) {
			int end = 0;

	 #pragma omp parallel default(none) shared(end, V) private(i,j)
			while ( !end ) {
				V[i][j] += 0.00001 + 0.0001 * (double)i/SIZE;
				if ( V[i][j] > 1.0 ) end = 1;
			}
		}

	/* Comprobación de resultados */
	double checksum = 0.0;
	for (i=0; i<SIZE; i++)
		for (j=0; j<SIZE; j++) {
			checksum = checksum + V[i][j] - 1;
		}

	printf("Resultado: %.10lf\n", checksum);
	return 0;
}
