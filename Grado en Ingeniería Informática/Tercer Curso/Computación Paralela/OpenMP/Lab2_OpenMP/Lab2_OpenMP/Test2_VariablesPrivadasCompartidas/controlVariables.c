#include<stdio.h>
/* 1. Include de biblioteca OpenMP */
#include <omp.h>
int main() {

	int v[ 10 ];
	int i;

	/* Inicializar */
	for ( i=0; i<10; i++ ) v[i] = 0;

	/* 2. Añadir directiva de paralelismo: Con vector v compartido */
	/* 3. Dentro de la región paralela: En una sóla línea de código,
	 *		cada thread escribe en la posición de su identificador
	 *		un 10 + su identificador
	 *		En la posición 0 acabará un 10, en la 1 un 11, etc...
	 */

	#pragma omp parallel shared(v)
	{	
		int id=omp_get_thread_num();
		if(id<10){
			v[id]=10+id;
		}
	}
	/* Se escribe el resultado en secuencial */
	for ( i=0; i<10; i++ ) printf( " %d", v[i] );
	printf( "\n" );

	return 0;
}
