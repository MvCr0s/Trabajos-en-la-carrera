import sys
import math

#Ejercicio1
numeroDNI=input()
numero=int(numeroDNI)
resto=numero%23
int(resto)
d={}
d[0]="T"
d[1]="R"
d[2]="W"
d[3]="A"
d[4]="G"
d[5]="M"
d[6]="Y"
d[7]="F"
d[8]="P"
d[9]="D"
d[10]="X"
d[11]="B"
d[12]="N"
d[13]="J"
d[14]="Z"
d[15]="S"
d[16]="Q"
d[17]="V"
d[18]="H"
d[19]="L"
d[20]="C"
d[21]="K"
d[22]="E"
print(d[resto])
