#
# Test de optimización manual
#

OBJS=matrixMult matrixMult2

FLAGS=-Wall -O3

todo: $(OBJS)

matrixMult: matrixMult.c
	gcc $(FLAGS) $^ -o $@

matrixMult2: matrixMult2.c
	gcc $(FLAGS) $^ -o $@


clean:
		rm -f $(OBJS)

