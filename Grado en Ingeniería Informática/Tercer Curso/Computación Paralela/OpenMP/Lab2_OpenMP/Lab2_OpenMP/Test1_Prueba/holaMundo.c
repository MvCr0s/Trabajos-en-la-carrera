#include<stdio.h>
/* 1. Include de biblioteca OpenMP */

#include <omp.h>

int main() {

	/* 2. Directiva de paralelismo */
	#pragma omp parallel
	{

	int id= omp_get_thread_num();
	int total = omp_get_num_threads();
	/* 3. Escribir el identificador de thread y la cantidad de threads */
	printf("Hola mundo desde el hilo %d de %d\n",id, total);
	}
	return 0;
}
