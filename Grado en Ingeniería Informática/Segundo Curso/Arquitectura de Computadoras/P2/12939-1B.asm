
.text
#a0 --> puntero a vector de indices
#a1 --> tamaño vector indices
#a2 --> puntero a vector
#a3 --> tamaño vector

#salida
#v0 --> suma elementos
#v1 --> error (0 --> todo ok, 1--> error ejecución)

SumaDispersa:
	li $t7,1
	#blt $a3, $a1, SumaDispersa_error  #si el vector de indices es mayor que el vector de datos, error
	slt $at, $a3, $a1
	li $t0, 0   #variable control bucle que recorre vector de indices
	li $v0, 0   #sumatorio
	beq $at, $t7,SumaDispersa_error 

Loop_SumaDispersa:
	lw $t1, 0($a0)	#cargo indice
	addi $t0, $t0,1 	#incrementamos variable puntero bucle (cambio)
	#ble $a3, $t1, SumaDispersa_error 	#si el tamaño del vector es menor o igual que el indice, error
	slt $at, $a3, $t1
	beq $a3, $t1, SumaDispersa_error
	sll $t1, $t1, 2  	# Calculamos dirección: index *4
	beq $at, $t7,SumaDispersa_error 
	addi $a0, $a0, 4	#incrementamos puntero	#Cargamos dato
	add $t1, $a2, $t1 	# Calculamos dirección: VEC + index*4
	
	lw  $t2, 0($t1)	
	
	add $v0, $v0, $t2
	beq  $t0, $a1, SumaDispersa_ok
	j Loop_SumaDispersa
	
SumaDispersa_ok:
	li $v1, 0
	jr $ra
	
SumaDispersa_error: 
	li $v1, 1
	jr $ra

    
#calcula máximo y mínimo de un vector 
#a0 --> puntero a vector
#a1 --> N
#devuelve
#v0 --> máximo
#v1 --> mínimo


MaxMin: 
	li $v0, 0x80000000  	#minimo de los enteros
	li $v1, 0x7FFFFFFF   	#maximo de los enteros
	li $t0, 0		#contador bucle
MaxMin_loop:
	lw $t1, 0($a0)		#CARGO PALABRA
	
	slt $at , $t1, $v0
	addi $t0, $t0, 1

	beq $at, $zero ,mover1
	j check_min
mover1:
	move $v0, $t1
check_min:
	#blt $v1, $t1, incr_punt 	#si v1 es menor que el valor leído, no actualizo el mínimo
	slt $at,$v1,$t1
	addi $a0, $a0, 4
	beq $at, $zero, mover2
	j incr_punt
mover2:
	move $v1, $t1
incr_punt:
	beq $t0, $a1, final
	j MaxMin_loop
final:
	jr $ra	
