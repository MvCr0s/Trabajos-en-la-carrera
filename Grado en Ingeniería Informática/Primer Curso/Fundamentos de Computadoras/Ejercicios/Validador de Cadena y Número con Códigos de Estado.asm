.data
mensaje: .asciiz "Dame una cadena: "
mensaje2: .asciiz "Dame un numero: "
cadena: .space 70


.text
main:
li $t1, 0 #conntador para ver si todo es correcto
li $t2,1
li $t3,2
li $t4,3
li $s0,0 #contador para ver la longitud de la cadena

#numero
move $s4, $v0   #pide la engtrada al usuario
li $v0,4
la $a0, mensaje2   #escribe el mensaje por pantalla 
syscall
li $v0, 5
syscall 
move $s5, $v0#pide la engtrada al usuario


#cadena
li $v0,4
la $a0, mensaje   #escribe el mensaje por pantalla 
syscall
li $v0, 8
la $a0,cadena
li $a1,70
syscall 

#calculo la longitud de la cadena
Longitud:
addu $s1,$a0,$s0
lbu $s2, 0($s1)
beqz $s2,Lon #cuando el puntero sea 0 quiere decir que ya se ha acabado la cadena
addi $s0,$s0,1
j Longitud
#compruebo si esta vacia e imprimo la long
Lon:
addi $s0,$s0,-1 #le resto uno al valor de la longitud para que de correctamente
bne $s0,$zero, Imprime #si en la primera posicion ha encontrado un 0 ya quiere decir que la cadena es nula
addi $t1,$t1,1
li $v0,1  #imprime t4
move $a0, $t4
syscall


Imprime:

#vemos si el numero es negativo
negativo:
blt $s5,$zero,menos
j cadenas


#vemos si el numero es mayor que la long cadena
cadenas:
blt $s0,$s5,menor
j Correcto

Correcto:
beq $t1,$zero,Bien #si t1 es cero quiere decir que es todo correcto
j fin #acabo

menos:
addi $t1,$t1,1 #incremento 1 al valor de t1 para indicar que ya no es todo correcto
li $v0,1  #imprime t2 
move $a0, $t2
syscall
j cadenas

menor:
addi $t1,$t1,1 #incremento 1 al valor de t1 para indicar que ya no es todo correcto
li $v0,1  #imprime t3
move $a0, $t3
syscall
j Correcto

Bien:
li $v0,1  #imprime t1
move $a0, $t1
syscall
j fin


fin:





