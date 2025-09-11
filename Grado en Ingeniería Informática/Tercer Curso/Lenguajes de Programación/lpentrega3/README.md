# LPEntrega3



## Objetivos

- Crear un analizador semántico con la comprobación de algunas reglas UML
- Traducción del modelo simplificado a un lenguaje objetivo 

## Análisis Semántico

El lenguaje a analizar incluye: 
    - Un modelo de clases UML
    - La declaración de restricciones OCL sobre el propio modelo

Partimos de una gramática ANTLR4 compuesta que importa una gramática para la especificación del modelo y otra gramática para una simplificación de OCL.
Ambas importan las definiciones del léxico común

## Estrategia 



## Traducción al lenguaje objetivo 

En caso de que no haya errores se generará el código Java que represente el modelo de clases. Todas las clases en un mismo archivo en el paquete default

## Estrategia
Dos opciones, representar el modelo UML en clases separadas o no usar un intermediario y que el Listener genere el código Java mientras recorre el árbol del Parser. 



# Autores

Ainhoa Carbajo Orgaz
Marcos de Diego Martín
Marcos Almaraz 
Daniel Ceballos

