
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#define SIZE	1024
#define SEED	6834723

int main() {
	int i,j,k;

	double A[ SIZE ][ SIZE ];
	double B[ SIZE ][ SIZE ];
	double C[ SIZE ][ SIZE ];

	srand48( SEED );
	
	for (i=0; i<SIZE; i++)
		for (j=0; j<SIZE; j++) {
			C[i][j] = 0;
			A[i][j] = drand48();
			B[i][j] = drand48();
		}
#pragma omp parallel for private(i,j,k) shared(A,B,C)
		for (i=0; i<SIZE; i++) 
for (k=0; k<SIZE; k=k+1) 
	for (j=0; j<SIZE; j++)
				C[i][j] = C[i][j] + A[i][k] * B[k][j];

	printf("Fin: %lf, %lf\n", C[0][0], C[SIZE-1][SIZE-1]);
	return 0;

}
