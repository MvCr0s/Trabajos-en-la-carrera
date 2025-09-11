analizador: aSintactico.tab.o lex.yy.o
	gcc aSintactico.tab.o lex.yy.o -o analizador -ll

aSintactico.tab.o: aSintactico.y
	bison -d aSintactico.y
	gcc -c aSintactico.tab.c -o aSintactico.tab.o

lex.yy.o: aLexico.l aSintactico.tab.h
	flex aLexico.l
	gcc -c lex.yy.c -o lex.yy.o

clean:
	rm -f analizador aSintactico.tab.c aSintactico.tab.h lex.yy.c *.o