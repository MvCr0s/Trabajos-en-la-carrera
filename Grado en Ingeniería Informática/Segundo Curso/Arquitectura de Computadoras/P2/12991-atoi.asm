atoi:
#Cogemos el primer caracter
lb $t1,0($a0)

#Inicializamos variables
li $t2,0
li $t3,10 #para ir multiplicando el numero
li $t4,-16 #cargo el valor del "0"-espacio  en ASCII
li $t5,48 #cargo el valor del 0 en ASCII
li $t9,45 #cargo el valor del -
li $t8,43 #cargo el valor del +
li $t6,-1	
li $v1,0
li $t7,9 #Para comparar que el numero este entre 0 y 9



beq $t1,$t9,signo #Comprobamos si empieza por un -
beq $t1,$t8,Siguiente
sub $t1,$t1,$t5
#Comprobamos que es un numero antes de entrar al bucle

#bltz  $t1, error1
slt $at, $zero, $t1
beq $t1,$t4,Siguiente 
bgt $t1,9, error1
bne $at, $zero, error1

Bucle:
add $t2,$t2,$t1
slt $at, $t2,$zero #Si la multiplicacion da negativa, hay overflow
bne $at,$zero, Overflow

Siguiente:
#comprobamos si hemos acabado
addi $a0,$a0,1
lb $t1,0($a0)
beq $t1, '\0' ,acabamos
#beq $t1,'-',signo #Comprobamos si empieza por un -
beq $t1,$t9,signo #Comprobamos si empieza por un -
sub $t1,$t1,$t5
beq $t1,$t4,Siguiente
bltz  $t1, acabamos
bgt $t1,$t7 , acabamos

mul $t2,$t2,$t3 #si no hemos acabado, lo multiplicamos x10 para sumar el siguiente digito
#blt $t2,$t6,Overflow #Si la multiplicacion da negativa, hay overflow
slt $at, $t2,$zero #Si la multiplicacion da negativa, hay overflow
bne $at,$zero, Overflow
j Bucle

error1:
li $v1,1
j fin

Overflow:
li $v1,2
j fin

signo:
addi $a0,$a0,1
lb $t1,0($a0)#comprobamos si lo siguiente es un numero
li $t0,48 #cargo un 48 para luego compara con t5 y asi ver si era negativo otro el numero
sub $t1,$t1,$t5
bgt $t1,$t7, error1
bltz  $t1, error1
add $t2,$t2,$t1
#li $t0,48 #cargo un 48 para luego compara con t5 y asi ver si era negativo otro el numero
j Siguiente

acabamos:
beq $t5,$t0,restar
fin:
move $v0,$t2
jr $ra


restar:
sub $t2,$zero,$t2
j fin

