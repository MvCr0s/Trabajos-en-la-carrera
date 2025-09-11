# LPEntrega2
Analizador sintáctico a partir de la Entrega 1


## Objetivos

 Analizador sintáctico basado en yacc/bison de archivos
 escritos en OCL, versión estándar 2.4.

## Estrategia
 Partir de la gramática. Analizar primero el lenguaje y definir las reglas.
 Una vez definidos las reglas y terminales se llama a Lex.
 Lex le pasa los tokens que tienen que cuadrar con las reglas.
 Por ejemplo: 
  Lex devuelve num ',' num ---> eso es un Real


La idea es que lex le pase a yacc yyval que tiene el valor léxico del token. 
Desaparecen las ocnstantes alfanuméricas.
## Autores
Ainhoa Carbajo orgaz

## Restricciones

- El analizador sintáctico deberá reportar error sintáctico indicando la línea del archivo de entrada
 en la que se ha detectado y el símbolo inesperado.
- Se definirá una constante MAX_SINTAX_ERROR que indicará el número máximo de
 errores sintácticos que se acumularán y reportarán sin detener el análisis.
-  Al finalizar el reconocimiento, y sólo en caso de que no se hayan producido errores sintácticos, el
 analizador desarrollado realizará la siguiente salida para cada clase distinta en la entrada:
        - nombre de la clase:
        - cantidad de invariantes de la clase:
        - cantidad de operaciones sobre colecciones utilizadas en los invariantes:
        - Operaciones de la clase especificadas: para cada operación de la clase especificada
            -nombre de la operación:
            -cantidad de operaciones sobre colecciones utilizadas en la especificación:
            -nombre de la operación:
            -cantidad de operaciones sobre colecciones utilizadas en la especificación
## Fecha LÍMITE
11/04/2025

