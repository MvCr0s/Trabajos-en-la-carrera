.data
MES: .word 31,28,31,30,31,30,31,31,30,31,30,31 #Vector año no bisiesto
BISIESTO: .word 31,29,31,30,31,30,31,31,30,31,30,31 #Vector año bisiesto
mensaje: .asciiz "Dame el número de un mes:"
mensaje2: .asciiz "Mes fuera de rango\n"
.text
main:
la $s0, MES
la $s1, BISIESTO
li $t3, 4
li $t5, 12
add $t1,$zero,$zero

Bucle:

li $v0,4
la $a0, mensaje#escribe el mensaje por pantalla 
syscall
li $v0, 5
syscall 
move $s3, $v0#pide la engtrada al usuario
beq $s3,$zero ,fin #Si el numero introducido es un 0 se acaba
addi $s3, $s3, -1 #Le restamos 1 para que coga la posicion correcta del array ya que empieza desde 0

loop:
blt $s3, $t5,salto #Si el numero introducido es mayor que 12, no corresponde con ningun mes por lo que tiene que introducir otro, si es correcto se continua
li $v0,5
la $a0, mensaje2#escribe el mensaje por pantalla 
syscall
j Bucle
salto:
div $t2,$t1,$t3#divido el contador entre 4
addi $t1,$t1, 1
mfhi $s4 	
bne $s4,$zero,NO #Solo cogemos del array bisiesto cada 4 (años) es decir cuando el resto del contador entre 4 sea 0 
mul $s3,$s3 , $t3 #Lo multiplicamos por 4 para ir a la posicion del array
add $s5,$s1,$s3
lw $t2, ($s5)
li $v0,1  #imprime t2 
move $a0, $t2
syscall
j Bucle 



NO: #Si no es año bisiesto cogemos los dias del array MES
mul $s3,$s3 , $t3
add $s5,$s0,$s3
lw $t2,($s5)
li $v0,1  #imprime t2 
move $a0, $t2
syscall
j Bucle 


fin:
