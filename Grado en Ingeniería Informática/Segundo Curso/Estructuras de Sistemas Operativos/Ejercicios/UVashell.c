#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<stdbool.h>
#include<unistd.h>
#include <sys/wait.h>

#define TAM 100

int main(int argc, char *argv[]){
    bool cont = true, cont1 = true;
    int i = 0;
    char *comando=NULL, *parametro1=NULL, *parametro2=NULL;
    char Buff[TAM+1], c, cwd[1024];
    comando = (char*) malloc ( sizeof (char)* TAM );
    parametro1 = (char*) malloc ( sizeof (char)* TAM );
    parametro2 = (char*) malloc ( sizeof (char)* TAM );
    char **comandoM = (char **)malloc(3 * sizeof(char *));

        //free(comando);
        //free(parametro1);
        //free(parametro2);

        for (int i = 0; i<=TAM; i++){Buff[i]=='\0';}

        fflush(stdin);
        i=0;
        cont1=true;

        while(cont){
                do{
                        read(0,&c,1);
                        if(c=='\n'){cont1=false;}
                        if (i>TAM){cont1=false;}
                        Buff[i] = c;
                        i++;
                }while(cont1);
                if (i>TAM){
                        write(1,"Exceso de tamaño\n",sizeof("Exceso de tamaño\n"));
                }
                Buff[i - 1] = '\0';
                i=0;
                while(Buff[i]!='\0' && Buff[i]!=32){ //cogemos el comando
                        comando[i]=Buff[i];
                        i++;
                }
                i++;
                if (strcmp(comando, "muestra") == 0){
                        for(int cogerParam=0; Buff[i]!='\0' && Buff[i]!=32;i++,cogerParam++){//char *parametro = Buff + 8; // Avanza 8 caracteres para obtener el parámetro
                                parametro1[cogerParam]=Buff[i];
                        }
                        if (strlen(parametro1) == 0) {
                                write(1,"Has usado mal el comando: muestra <archivo>.\n",sizeof("Has usado mal el comando: muestra <archivo>.\n"));
                        } else {
                                 pid_t pid = fork();

                                if (pid == -1) {
                                        write(1,"Error al crear el proceso hijo.\n",sizeof("Error al crear el proceso hijo.\n"));
                                } else if (pid == 0) {
                                        if (access(parametro1, F_OK) == -1) {
                                                write(1,"Error. El archivo no exite.\n",sizeof("Error. El archivo no exite.\n"));
                                        } else {
                                        // Construir dinámicamente el arreglo de parámetros
                                        comandoM[0] = "/bin/cat";
                                        comandoM[1] = parametro1;
                                        comandoM[2] = NULL;

                                        execvp("/bin/cat", comandoM);

                                        free(comando);
                                        free(parametro1);

                                        // Si execvp() vuelve, ha habido un error
                                        write(1,"Error ejecutando 'cat' \n",sizeof("Error ejecutando 'cat' \n"));

                                        }

                                        exit(0);
                                } else {
                                        int status;
                                        wait(&status);

                                }
                        }

                }


                if ((strcmp(comando, "copia") == 0)){
                        for(int cogerParam=0; Buff[i]!='\0' && Buff[i]!=32;i++,cogerParam++){
                                parametro1[cogerParam]=Buff[i];
                        }
                        i++;
                        for(int cogerParam=0; Buff[i]!='\0' && Buff[i]!=32;i++,cogerParam++){
                                parametro2[cogerParam]=Buff[i];
                        }
                        if ((strlen(parametro1) == 0) ||(strlen(parametro2)==0)) {
                                write(1,"Has usado mal el comando: copia <archivo> <archivo>\n",sizeof("Has usado mal el comando: copia <archivo> <archivo>\n"));
                        } else {
                                pid_t pid = fork();
                                if (pid == -1) {
                                        write(1,"Error al crear el proceso hijo\n", sizeof("Error al crear el proceso hijo\n"));
                                } else if (pid == 0) {
                                        if (access(parametro1, F_OK) == -1 || (access(parametro2, F_OK) == 0)) {
                                                write(1,"Error. El archivo ay existe.\n",sizeof("Error. El archivo ay existe.\n"));
                                        } else {
                                                // Construir dinámicamente el arreglo de parámetros
                                                comandoM[0] = "./copia";
                                                comandoM[1] = parametro1;
                                                comandoM[2] = parametro2;
                                                comandoM[3] = NULL;
                                                execvp("./copia", comandoM);

                                                free(comando);
                                                free(parametro1);
                                                free(parametro2);

                                                // Si execvp() vuelve, ha habido un error
                                                write(1,"Error ejecutando 'copia' \n",sizeof("Error ejecutando 'copia' \n"));
                                                        }

                                        exit(0);
                                } else {
                                        int status;
                                        wait(&status);

                                }
                        }

                }
                if ((strcmp(comando,"lista")==0)){
                        for(int cogerParam=0; Buff[i]!='\0' && Buff[i]!=32;i++,cogerParam++){
                                parametro1[cogerParam]=Buff[i];
                        }
                        pid_t pid = fork();
                        if (pid == -1){
                                write(1,"Error al crear el proceso hijo\n", sizeof("Error al crear el proceso hijo\n"));
                        }else if (pid == 0) {
                                if (Buff[i++] == '\0'){
                                        if((strlen(parametro2) == 0)){
                                                 // Construir dinámicamente el arreglo de parámetros
                                                comandoM[0] = "/bin/ls";
                                                if((strlen(parametro1)!=0)){comandoM[1] = parametro1;}
                                                else{comandoM[1] = getcwd(cwd, sizeof(cwd));}
                                                comandoM[2] = NULL;

                                                execvp("/bin/ls", comandoM);

                                                // Si execvp() vuelve, ha habido un error
                                                write(1,"Error ejecutando 'ls' \n",sizeof("Error ejecutando 'ls' \n"));
                                        }
                                        free(comando);
                                        free(parametro1);
                                        exit(0);
                                        }
                                        else{
                                                write(1,"Error. El uso del comando es: lista o lista <directorio>.",sizeof("Error. El uso del comando es: lista o lista <directorio>."));
                                        }
                                } else {
                                int status;
                                wait(&status);
                        }
                }
                if ((strcmp(comando, "salir")==0)){
                        free(comandoM);
                        return 1;
                        }
                for (int j = 0; j<=i;j++){
                        Buff[j]='\0';
                }
                comando[0] = '\0';
                comando[5] = '\0';
                comando[6] = '\0';
                parametro1[0] = '\0';
                parametro2[0] = '\0';
                free(comandoM[0]);
                free(comandoM[1]);
                fflush(stdin);
                i = 0;
                cont1=true;
        }
}
