#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include <semaphore.h>

typedef struct listaenlazada consumoData;

typedef struct {
    char tipo;
    int proveedorID;
} Producto;


typedef struct {
        char path[1000];
        int id;
        int T;
        Producto *buffer;
        char archivo_destino[1000];
}parametrosProveedor;

typedef struct {
        int id;
        int T;
        Producto *buffer;
}parametrosConsumidor;

typedef struct listaenlazada {
        int consumidor;
        int productostotales[10];
        int proveedores[10];
        consumoData *sig;
}consumoData;


int consumidoresTerminados = 0;
int C,P,T;
int proveedoresPorTerminar;
int indiceP = 0;
int indiceC = 0;
int salirBucleConsumidores = 0;
consumoData *listaenlazada = NULL;
consumoData *Princ = NULL; //principio de la lista



sem_t hayEspacio;
sem_t hayDatos;
sem_t mutex_buffer;
sem_t mutex_arch_prov;
sem_t mutex_proveeTerminados;
sem_t mutex_indiceProv;
sem_t mutex_indiceCons;
sem_t mutex_lista;
sem_t facturador_on;

int esnum(char *cad) {
    for (int j = 0; j < strlen(cad); j++) {
        if (cad[j] < '0' || cad[j] > '9') { return 0; }
    }
    return 1;
}

//------------------------------------------------------------------------------------------------------

void *Proveedor(void *arg) {
    parametrosProveedor *param = (parametrosProveedor *)arg;
    char caracter;
    int index = 0;
    int productosLeidos = 0;
    int productosInvalidosProveedor = 0;
    int productosValidos = 0;
    int productosPorTipo[10] = {0}; // Array para contar productos de cada tipo valido

    int TamBuffer = param->T;
    int id = param->id;
    //abrimos el fichero con el path
    FILE *archivoEntrada = fopen(param->path, "r");

    if (archivoEntrada == NULL) {
        perror("Error al abrir el archivo de entrada");
        exit(EXIT_FAILURE);
    }

    while (fscanf(archivoEntrada, " %c", &caracter) == 1) { //leemos el fichero

        if ('a' <= caracter && caracter <= 'j') {

            sem_wait(&hayEspacio);
            //recorremos el buffer rellenandolo con cada caracter(tipo)
            //y el proveedor que lo ha rellenado(id)
            sem_wait(&mutex_indiceProv);
            index = indiceP;
            indiceP = (indiceP + 1) % TamBuffer;//Actualizamos el indice
            sem_post(&mutex_indiceProv);
            sem_post(&mutex_buffer);
            param->buffer[index].proveedorID = id;
            param->buffer[index].tipo = caracter;
            sem_wait(&mutex_buffer);
            sem_post(&hayDatos);

            productosPorTipo[caracter - 'a']++; // Contabilizamos cuantos caracteres hay
            //Actualizamos contadores
            productosValidos++;
        } else {
            productosInvalidosProveedor++;
        }
        productosLeidos++;
    }

    fclose(archivoEntrada);
    //modificamos el struct de los parametros
    sem_wait(&mutex_arch_prov);
    FILE *archivo_destino = fopen(param->archivo_destino, "a");
    if (archivo_destino == NULL) {
        perror("Error al abrir el archivo de entrada");
        exit(EXIT_FAILURE);
    }

    //Escribir identificador del proveedor
    fprintf(archivo_destino, "Proveedor %d:\n", (param->id));
    //Escribir nº de productos totales                  
    fprintf(archivo_destino, "\tProductos leidos: %d\n",productosLeidos); 
    //Escribir nº de productos validos                  
    fprintf(archivo_destino, "\tProductos válidos: %d\n", productosValidos);  
    //Escribir nº de productos invalidos              
    fprintf(archivo_destino, "\tProductos inválidos: %d\n", productosInvalidosProveedor);            

    //Escribir cantidad de productos por tipo
    for (int i = 0; i < 10; i++) {
        fprintf(archivo_destino, "\t\tNúmero de productos de tipo %c: %d\n", 'a' + i, productosPorTipo[i]);
    }

    fprintf(archivo_destino, "\n");
    fclose(archivo_destino);
    sem_post(&mutex_arch_prov);
    //contador para saber lo proveedores que han terminado
    sem_wait(&mutex_proveeTerminados);
    proveedoresPorTerminar--;
    sem_post(&mutex_proveeTerminados);
    //Cuando finalicen todos los proveedores añadiremos una 'F' como
    //caracter y un -1 como id para señalizar
    //en el buffer que es el final
    if(proveedoresPorTerminar==0){
        sem_wait(&hayEspacio);
        param->buffer[(index+1)%TamBuffer].proveedorID = -1;
        param->buffer[(index+1)%TamBuffer].tipo ='F';
        sem_post(&hayDatos);
    }
    pthread_exit(NULL);
}

//------------------------------------------------------------------------------------------------------

void *ClienteConsumidor(void *arg) {
    parametrosConsumidor *param2 = (parametrosConsumidor *)arg;
    int index = 0;
    int productosConsumidos = 0;
    int productosPorTipo[10] = {0};//array para contabilizar cuantos productos hay de cada caracter
    int productosPorProveedor[10] = {0};//array para contabilizar todos los productos de cada proveedor
    Producto pConsumido;
    int TamBuffer = param2->T;
    while (salirBucleConsumidores==0) {
        
        sem_wait(&hayDatos);
        sem_wait(&mutex_indiceCons);
        index = indiceC;
        indiceC=(indiceC + 1) % TamBuffer;//actualizamos el indice
        sem_post(&mutex_indiceCons);
        //consumimos el buffer
        pConsumido = param2->buffer[index];
        sem_post(&hayEspacio);
        //comprobamos si hemos llegado al final del buffer comparando con el caracter 'F' e id -1 que añadimos al final
        if(pConsumido.tipo=='F' || pConsumido.proveedorID==-1){
                sem_wait(&mutex_indiceCons);
                indiceC=(indiceC - 1) % TamBuffer;
                sem_post(&mutex_indiceCons);
                sem_post(&hayDatos);
                sem_post(&hayEspacio);
                salirBucleConsumidores = 1;
        }else{//si no hemos llegado al final, actualizamos los contadores
                productosPorTipo[pConsumido.tipo-'a']++;
                productosPorProveedor[pConsumido.proveedorID]++;
                productosConsumidos++;
        }
    }
    // Actualizar la lista de consumo con los datos del consumidor actual
    sem_wait(&mutex_lista);
    consumoData *nuevoNodo = (consumoData*)malloc(sizeof(consumoData));
    if (nuevoNodo == NULL) { exit(EXIT_FAILURE); }
    nuevoNodo->consumidor = param2->id;
    for (int i = 0; i < 10; i++) {
        nuevoNodo->productostotales[i] = productosPorTipo[i];
        nuevoNodo->proveedores[i] = productosPorProveedor[i];
        nuevoNodo->consumidor = param2->id;
    }
    nuevoNodo->sig = listaenlazada;
    listaenlazada = nuevoNodo;
    sem_post(&mutex_lista);
    sem_post(&facturador_on);
    

    pthread_exit(NULL);
}

//------------------------------------------------------------------------------------------------------

void *funcionFacturador(void *arg) {
    char *archivo = (char *)arg;
    int Terminados = 0;
    int productosConsumidos = 0;
    int total_prod = 0;
    int total_por_productor [P];
    int total_por_consumidor [C];
    int max = 0;
    int consumidor_max = 0;
    FILE *archivo_destino = fopen(archivo, "a");
    if (archivo_destino == NULL) {
        perror("Error al abrir/crear el fichero de salida");
        pthread_exit(NULL);
    }

    // Esperar a que todos los consumidores hayan terminado
    while (Terminados < C) {
        productosConsumidos = 0;
        sem_wait(&facturador_on);
        sem_wait(&mutex_lista);
        consumoData *actual = listaenlazada;
        // Verificar que la lista no esté vacía antes de procesar
        if (actual != NULL) {
                fprintf(archivo_destino, "Cliente consumidor %d:\n", actual->consumidor);
                for (int i = 0; i < 10; i++) {
                        productosConsumidos += actual->productostotales[i];
                }
                total_por_consumidor[actual->consumidor]=productosConsumidos;
                total_prod = total_prod + productosConsumidos;
                fprintf(archivo_destino, "\tProductos consumidos: %d\n", productosConsumidos);
                for (int i = 0; i < 10; i++) {
                        fprintf(archivo_destino, "\t\tProducto tipo '%c': %i\n", 'a' + i, actual->productostotales[i]);
                }
                for(int i = 0; i<P;i++){
                        total_por_productor[i]=total_por_productor[i]+actual->proveedores[i];
                }
                fprintf(archivo_destino, "\n");
                listaenlazada = actual->sig; // Avanzar al siguiente nodo
                free(actual); // Liberar el nodo procesado
        }
        sem_post(&mutex_lista);
        Terminados++;
        }

        fprintf(archivo_destino,"Total de productos consumidos: %i.\n",total_prod);
        for(int i = 0; i<P;i++){
                fprintf(archivo_destino,"\t%i del proovedor %i.\n",total_por_productor[i],i);
         }

        max = total_por_consumidor[0];
        consumidor_max = 0;
        for(int i = 1; i<C;i++){
                if(total_por_consumidor[i]>max){
                        consumidor_max = i;
                }
        }

        fprintf(archivo_destino,"Cliente consumidor que mas ha consumido: %i.\n",consumidor_max);

        fclose(archivo_destino);
        pthread_exit(NULL);
}
        
//------------------------------------------------------------------------------------------------------

int main(int argc, char *argv[]){
        FILE *archivo_destino;
        bool BT = true, BP = true, BC = true;
        int esnum(char *cad);
        int consumidores=0;
        parametrosProveedor *paramProveedor;
        parametrosConsumidor *paramConsumidor;
        Producto* buffer = NULL;
        pthread_t* proveedores = NULL;
        pthread_t* Consumidores = NULL;
        pthread_t* hiloFacturador = NULL;

        //Comprobamos q los parametros sean los correctos
        if (argc != 6){
                write(1,"Uso erroneo. El uso del programa es: ./<program> <path> <outputFile> <T> <P> <C>\n",82);
                return 1;
        }

        if(esnum(argv[3])==0){write(1,"El argumento de Buffer debe ser un numero.\n",44);return 1;}
        if(esnum(argv[4])==0){write(1,"El argumento de Proovedores debe ser un numero.\n",49);return 1;}
        if(esnum(argv[5])==0){write(1,"El argumento de Consumidores debe ser un numero.\n",50);return 1;}

        T = atoi(argv[3]);
        P = atoi(argv[4]);
        proveedoresPorTerminar = atoi(argv[4]);
        C = atoi(argv[5]);

        if((T>5000 || T<1)){
                write(1,"El tamaño del Buffer debe de estar entre 1 y 5000. \n",49);
                return 1;
        }

        if((P>7 || P<1)){
                write(1,"El numero de proovedores debe estar entre 1 y 7. \n",51);
                return 1;
        }

        if((C>1000 || C<1)){
                write(1,"El numero de clientes consumidores debe estar entre 1 y 1000. \n", 50);
                return 1;
        }

        //Inicializamos las estructuras como memoria dinamica y comprobamos que se hayan inicializado correctamemente
        buffer = (Producto *)malloc(sizeof(Producto) * T);
        if (buffer == NULL) { exit(1); }
        proveedores = (pthread_t *)malloc(sizeof(pthread_t) * P);
        if (proveedores == NULL) { exit(1); }
        paramProveedor = (parametrosProveedor *) malloc(sizeof(parametrosProveedor) * P);
        if (paramProveedor == NULL) { exit(1); }
        paramConsumidor = (parametrosConsumidor *) malloc(sizeof(parametrosConsumidor) * C);
        if (paramConsumidor == NULL) { exit(1); }
        Consumidores = (pthread_t *)malloc(sizeof(pthread_t) * C);
        if (Consumidores == NULL){exit(1);}
        hiloFacturador = (pthread_t *) malloc(sizeof(pthread_t));
        if (hiloFacturador == NULL){exit(1);}


        if(sem_init(&hayEspacio,0,  T)==-1){
                perror("Error al inicializar el semáforo");
                exit(-1);
        }
        if(sem_init(&hayDatos,0,  0)==-1){
                perror("Error al iniciar el semaforo");
                exit(-1);
        }
        if(sem_init(&mutex_buffer,0,  1)==-1){
                perror("Error al iniciar el semaforo");
                exit(-1);
        }
        if(sem_init(&mutex_indiceProv,0,  1)==-1){
                perror("Error al iniciar el semaforo");
                exit(-1);
        }
        if(sem_init(&mutex_indiceCons,0,  1)==-1){
                perror("Error al iniciar el semaforo");
                exit(-1);
        }
        if(sem_init(&mutex_proveeTerminados,0,  1)==-1){
                perror("Error al iniciar el semaforo");
                exit(-1);
        }
        if(sem_init(&mutex_arch_prov,0,  1)==-1){
                perror("Error al iniciar el semaforo");
                exit(-1);
        }
        if(sem_init(&mutex_lista,0,  1)==-1){
                perror("Error al iniciar el semaforo");
                exit(-1);
        }
        if(sem_init(&facturador_on,0, 0)==-1){
                perror("Error al iniciar el semaforo");
                exit(-1);
        }

        //Creamos los hilos de proveedores con sus parametros correspondientes
        for (int i = 0; i < P; i++) {
                paramProveedor[i].id = i;  // Asigna directamente el identificador del proveedor
                sprintf(paramProveedor[i].path, "%s/proveedor%i.dat", argv[1], i);
                paramProveedor[i].T = T;
                paramProveedor[i].buffer = buffer;
                strcpy(paramProveedor[i].archivo_destino, argv[2]);

                if (pthread_create(&proveedores[i], NULL, Proveedor, (void *)&(paramProveedor[i])) != 0) {
                        perror("Error al crear hilo proveedor");
                        return 1;
                }
        }
        //Creamos los hilos consumidores con sus parametros correspondientes
        for (int i = 0; i < C; i++) {
                paramConsumidor[i].id = i;
                paramConsumidor[i].T = T;
                paramConsumidor[i].buffer = buffer;
                if(pthread_create(&Consumidores[i], NULL, ClienteConsumidor, (void *)&(paramConsumidor[i]))!= 0){return 1;}
        }
        //Creamos el hilo de facturador
        if (pthread_create(hiloFacturador, NULL, funcionFacturador, argv[2]) != 0) {
                perror("Error al crear hilo facturador");
                return 1;
        }

        for (int i = 0; i < P; i++) {
                pthread_join(proveedores[i], NULL);
        }

        for (int i = 0; i < C; i++) {
                pthread_join(Consumidores[i], NULL);
        }

        pthread_join(*hiloFacturador, NULL);

        // Libera la memoria de los hilos
        free(buffer);
        free(proveedores);
        free(Consumidores);
        free(paramConsumidor);
        free(paramProveedor);
        free(hiloFacturador);
        free(listaenlazada);

        //Eliminar semáforos
        sem_destroy(&hayEspacio);
        sem_destroy(&hayDatos);
        sem_destroy(&mutex_buffer);
        sem_destroy(&mutex_indiceProv);
        sem_destroy(&mutex_indiceCons);
        sem_destroy(&mutex_proveeTerminados);
        sem_destroy(&mutex_arch_prov);
        sem_destroy(&mutex_lista);
        sem_destroy(&facturador_on);

        // Asegura que el hilo principal no termine antes que los hilos secundarios
        pthread_exit(NULL);
        return 0;
}
