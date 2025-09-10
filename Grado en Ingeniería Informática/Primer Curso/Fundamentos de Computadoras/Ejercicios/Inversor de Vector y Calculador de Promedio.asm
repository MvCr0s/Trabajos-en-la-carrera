.data
B: .word 0:8
A: .word 1, 2, 14, 18, 26, 23, 35, 65 # Declaración del vector

.text
main: 
la $s0 ,A 
la $s1 , B
li $s5, 8#s5 es la longuitud del vectorz

 

addi $s0 , $s0 , 28 
add $t1 , $zero , $zero
add $s3 , $zero , $zero
 Bucle: 
 
 lw $t0, ($s0)
 sw $t0, ($s1)  
 addi $s0, $s0 , -4 #resta a s0 p 4
 addi $s1, $s1 ,4 #suma a s1  4
 addi $t1 ,$t1 , 1 # incretmento el contador
 add $s3 , $s3 , $t0# sumo a s3 los valores del vecotr b
 bne $t1 , $s5 , Bucle
 
 div $s3, $s3 , $s5
