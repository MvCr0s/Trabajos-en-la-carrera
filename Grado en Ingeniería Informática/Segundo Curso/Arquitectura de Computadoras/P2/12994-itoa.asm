itoa:
.data
str: .space 32

.text
beq $a0, $zero, fin
li $t0,48 
li $t1,10 
la $t2, str 
move $t6,$a0

empezamos:
beqz $a1,ComplementoCero
beq $a1,1,BinarioCero
beq $a1,2,ComplementoUno
beq $a1,3,BinarioUno
beq $a1,4,ComplementoDos
beq $a1,5,BinarioDos

ComplementoCero:
blt $t6,$zero,NegativoC2
j CogerNum0

NegativoC2:
sub $t6,$zero,$t6
li $t9,48



CogerNum0:
beq $t6, '\0' ,acabamos
divu $t6,$t1
mfhi $t4
mflo $t6
addi $t4,$t4 ,48
sb $t4,0($t2)
addi $t2,$t2,1
j CogerNum0


BinarioCero:
andi $t6,$t6,0xffffffff
j CogerNum0

ComplementoUno:
sll $t6,$t6,16
blt $t6,$zero,Negativobin16
srl $t6,$t6,16
j CogerNum0

Negativobin16:
sub $t6,$zero,$t6
srl $t6,$t6,16
li $t9,48
sub $t6,$zero,$t6
j CogerNum0

BinarioUno:
blt $t6,$zero, NegativoDieciseis
j CogerNum0

NegativoDieciseis:
andi $t6, $t6, 0x0000ffff
j CogerNum0


ComplementoDos:
sll $t6,$t6,24
blt $t6,$zero,Negativobin8
srl $t6,$t6,24
j CogerNum0

Negativobin8:
sub $t6,$zero,$t6
li $t9,48
srl $t6,$t6,24
j CogerNum0

BinarioDos:
blt $t6,$zero, NegativoOcho
j CogerNum0

NegativoOcho:
andi $t6, $t6, 0x000000ff
j CogerNum0

acabamos:
beq $t9,48,X
restauramosCadena:
addi $t8,$t5,0
addi $t2, $t2,-1
j loop

X:
li $t7,45
sb $t7,($t2)
addi $t5,$t5,1
addi $t2,$t2,1
j restauramosCadena


loop:
beq $t5, 0 ,fin
lb $t7, ($t2)
addi $t5,$t5,-1
addi $t2,$t2,-1
sb $t7,($a2)
addi $a2,$a2,1
j loop


fin:
sb $zero,($a2)
jr $ra
