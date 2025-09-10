#Este código MIPS le pide al usuario un número entero de 32 bits y luego interpreta el patrón de bits de ese número como si fuera un número de punto flotante de precisión simple (según el estándar IEEE 754).
#El programa descompone ese patrón de bits en sus tres partes fundamentales y las muestra por pantalla:
#Signo: Determina si el número es positivo o negativo.
#Característica (Exponente): Extrae los 8 bits del exponente y calcula su valor real restando el sesgo (bias).
#Fracción (Mantisa): Extrae los 23 bits de la parte fraccionaria.


.data
mensaje:.asciiz "Dame un número entero de 32 bits\n"
mensaje_Signo:.asciiz "Signo: "
mensaje_Fraccion:.asciiz "\nFraccion: "
mensaje_Carac:.asciiz "Caracteristica: "
positivo: .asciiz "positivo\n"
negativo: .asciiz "negativo\n"
.text
main:
    #x
    li $v0,4
    la $a0, mensaje   #escribe el mensaje por pantalla 
    syscall
    li $v0, 5
    syscall 
    move $s0, $v0 #pide la engtrada al usuario y la guarda en $s0
    j signo
    
signo:
    li $s2, 32
    li $s1, 30	
   
    li $v0,4
    la $a0, mensaje_Signo   #escribe el mensaje por pantalla 
    syscall
    
    jal ExtraeBits
    
    # Recuperamos el resultado en $v1 y el código de error en $v0
    move $t1, $v0
    
    beq $t1, $zero, es_positivo   # si $t1 es cero, saltar a etiqueta es_positivo
    # si no, es negativo
    li $v0, 4       # cargamos el código de servicio 4 para imprimir cadena
    la $a0, negativo    # cargamos la dirección de memoria de la cadena negativo
    syscall             # imprimimos la cadena
    j Caracteristica               # saltamos a la etiqueta fin

    es_positivo:
    li $v0, 4       # cargamos el código de servicio 4 para imprimir cadena
    la $a0, positivo    # cargamos la dirección de memoria de la cadena positivo
    syscall             # imprimimos la cadena
    
    j Caracteristica
    
Caracteristica:
    li $s2, 30
    li $s1, 23
    
    li $v0,4
    la $a0, mensaje_Carac   #escribe el mensaje por pantalla 
    syscall
    
    jal ExtraeBits
 
    # Recuperamos el resultado en $v1 y el código de error en $v0
    move $t1, $v0
    addi  $t1,$t1,-127
     # Imprimimos el resultado
    li $v0, 1
    move $a0, $t1
    syscall
    
    j Fraccion
    
Fraccion:    
    li $s2, 22
    li $s1, 0	
    
    li $v0,4
    la $a0, mensaje_Fraccion   #escribe el mensaje por pantalla 
    syscall
    
    jal ExtraeBits
    
    # Recuperamos el resultado en $v1 y el código de error en $v0
    move $t1, $v0
    
     # Imprimimos el resultado
    li $v0, 1
    move $a0, $t1
    syscall
    
    li $v0, 10
    syscall
    
    

    
    
    ExtraeBits: 
    # Calcular la longitud de la cadena binaria de x
    addi $s3,$s0,0
    li $t0, 32       # Cargar 32 en $t0
    clz $t1, $s3     # Contar los ceros a la izquierda de x y guardar el resultado en $t1
    sub $t0, $t0, $t1 # Restar la cantidad de ceros a la izquierda de x a 32
                 # El resultado en $t0 es la longitud de la cadena binaria de x

   # Calcular la longitud del subconjunto de bits
   sub $t3, $s2, $s1 # Calcular la cantidad de bits en el subconjunto
   addi $t3, $t3, 1  # Añadir 1 para incluir el bit de orden m
                 # El resultado en $t3 es la longitud del subconjunto de bits

   # Calcular la máscara para el subconjunto de bits
   li $t4, 1        # Cargar 1 en $t4
   sllv $t4, $t4, $t3 # Desplazar 1 a la izquierda por la longitud del subconjunto de bits
   addi $t4, $t4, -1  # Restar 1 para obtener una máscara de bits de longitud del subconjunto de bits

   # Mover el subconjunto de bits de x a la derecha
   srlv $s3, $s3, $s1 # Desplazar x a la derecha por el número de bits de orden n
   and $s3, $s3, $t4 # Aplicar la máscara para obtener solo el subconjunto de bits

   # Guardar el resultado en $v1
   move $v0, $s3     # Mover el resultado a $v1

    # Salimos de la función3
    jr $ra
    
    
