
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <omp.h>

#define SIZE	512
#define SEED	6834723

int main() {
	int i,j,k;

	double A[ SIZE ][ SIZE ];
	double B[ SIZE ][ SIZE ];
	double C[ SIZE ][ SIZE ];
	double D[ SIZE ][ SIZE ];

	srand48( SEED );

	for (i=0; i<SIZE; i++)
		for (j=0; j<SIZE; j++) {
			C[i][j] = 0;
			D[i][j] = 0;
			A[i][j] = drand48();
			B[i][j] = drand48();
		}

	/* Paralelizar esta versión de cada una de las 6 posibles formas */
	for (i=0; i<SIZE; i++)
		#pragma omp parallel for shared (A,B,C), private(k,j)
		for (k=0; k<SIZE; k++)
			for (j=0; j<SIZE; j++)
			#pragma omp atomic
				C[i][j] = C[i][j] + A[i][k] * B[k][j];
	/* NO PARALELIZAR: Versión secuencial para comprobar resultados */
	for (i=0; i<SIZE; i++)
		for (j=0; j<SIZE; j++) 
			for (k=0; k<SIZE; k++) 
				D[i][j] = D[i][j] + A[i][k] * B[k][j];

	/* Comprobación de resultados */
	int correcto = 1;
	int casi_correcto = 1;
	for (i=0; i<SIZE && correcto; i++)
		for (j=0; j<SIZE && correcto; j++) {
			if ( C[i][j] != D[i][j] ) correcto = 0;
			if ( fabs( C[i][j] - D[i][j] ) > 0.0000001 ) casi_correcto = 0;
		}

	printf("Correcto:      ");
	if ( correcto )
		printf("OK\n");
	else
		printf("ERROR!!\n");

	printf("Casi correcto: ");
	if ( casi_correcto )
		printf("OK\n");
	else
		printf("ERROR!!\n");

	return 0;
}
