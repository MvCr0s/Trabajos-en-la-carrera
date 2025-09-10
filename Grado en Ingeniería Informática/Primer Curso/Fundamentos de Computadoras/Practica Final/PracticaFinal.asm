.data
A: .word 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 #matriz A todo en esta linea  Para acceder a un elemento  A+4*i*n(tamaño de la matriz)+4*j
B: .space 64
#indice = fila * num_columnas + columna
#La matriz A no puede ser 1*1 (si 1*6,6*1) ni mayor que 8*8 
#La norma lo que hace es sumar todos los elementos de cada columna y se queda con el mayor
.text   
main:
la $a0,A
li $a1,4 #filas A  (i) ----->m
addi $a1,$a1,-1
sll $a1,$a1,4
li $a2,6 #columnas A   (j) ------>n
addi $a2,$a2,-1
sll $a2,$a2,4
li $a3,3 #1<=N<=10



beq $v0,1,printMatrix


stencil:  
#apilar      
la $s7,B    
addi $sp,$sp,-20
sw $ra,0($sp)
sw $s0,4($sp)
sw $s1,8($sp)
sw $s2,12($sp)
sw $s3,16($sp)
move $s0,$a0#A
move $s1,$a1#filas
addi $s1,$s1,-1
sll $s1,$s1,4
move $s2,$a2#columns
addi $s2,$s2,-1
sll $s2,$s2,4
move $s3,$a3#N

contadores:
li $s4,0  #contador de filas
li $s5,0  #contandor de columnas
li $s6,1  #contador de cuantas veces
#codigo
blt $s3,1,e2

recorrer:
beq $s5,$s2,cambioFila
addi $s5,$s5,4
j loop

cambioFila:
beq $s1,$s4,mirocol
addi $s4,$s4,4  #contador de filas
li $s5,0
j loop
mirocol:
beq $s2,$s5,reset
addi $s5,$s5,4  #contador de filas
j loop
reset:
addi $s6,$s6,1 
#indice = fila * num_columnas + columna

#Copiams la matriz B en A
copiarMatriz:
addi $s7,$s4,-4         
sllv $s7,$s7,$s2
add $s7,$s7,$s5
la $s0,$s7($s7)
j RECORRER

RECORRER:
beq $s5,$s2,cambioFila
addi $s5,$s5,4
j copiarMatriz:

CAMBIOFILA:
beq $s1,$s4,MIROCOL
addi $s4,$s4,4  #contador de filas
li $s5,0
j copiarMatriz

MIROCOL:
beq $s2,$s5,RESET
addi $s5,$s5,4  #contador de filas
j copiarMatriz

RESET:
li $s4,0  #contador de filas
li $s5,0  #contandor de columnas
#Copiamos la matriz B en A para hacerlo N veces
blt $s6,$s3,loop
j norma


loop:
#vecino de arriba                #move $a0,  vecino de arriba
addi $a0,$s4,-4         
sllv $a0,$a0,$s2
add $a0,$a0,$s5
#vecino abajo
addi $a1,$s4,4          
sllv  $a1,$a1,$s2
add $a1,$a1,$s5    #move $a1,   vecino de debajo
#vecino izquierda
addi $a2,$s5,-4         
sllv $a2,$a2,$s2	 #move $a2,  vecino de la izq
add $a2,$a2,$s5 
#vecino derecha
addi $a3,$s5,4         
sllv $a3,$a3,$s2
add $a3,$a3,$s5                  #move $a3,   vecino de la drcha


#1 vecino(array),2 vecinos,3 vecinos,4 vecinos
bnez $s1,unVecino
unVecino:
jal ponderacion1

bnez $s4,dos0tres
beq $s4,$s1,dos0tres
j tres0cuatro

dos0tres:
bnez $s5,dosVecinos
beq $s5,$s2,dosVecinos
j tresVecinos

tres0cuatro:
beq $s5,$s2,tresVecinos
beq $s4,$s1,tresVecinos
j cuatroVecinos

dosVecinos:  
jal ponderacion2

tresVecinos:
jal ponderacion3 

cuatroVecinos:
jal ponderacion4    
#con el e1 igual
j recorrer
norma:
jal norma
j desapilar


e2:
li $v0,2
j desapilar

desapilar:
lw $s0,4($sp)
lw $s1,8($sp)
lw $s2,12($sp)
lw $s3,16($sp)
jr $ra
#El numero que calculamos de aqui es el que colocamos en la casilla(podemos crear otras matriz y colocarlo ahi pero $a0 tiene que acabar apuntando a la nueva matriz(Preocuparse de $a0 cuando la apomderacion sea impar))



#Para k=1....N
	#Para i=0..... m-1
		#para j=0......n-1
			#-Obtengo los vecinos
			#-LLamo a ponderacionX
			#-Almaceno resultado($v0)en Matriz B
		#finPara
	#finPara
	#jal PrintMatriz
#finPara
#jal Norma--->$a0
#Desapilar
#jr $ra

