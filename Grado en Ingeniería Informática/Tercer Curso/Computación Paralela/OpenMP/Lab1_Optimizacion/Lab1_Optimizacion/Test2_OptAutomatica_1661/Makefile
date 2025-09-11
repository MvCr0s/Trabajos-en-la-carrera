#
# Test de optimización automática
#

OBJS=matrixMult_O0 matrixMult_dyn_O0 matrixMult_O3 matrixMult_dyn_O3

FLAGS=-Wall

todo: $(OBJS)

matrixMult_O0: matrixMult.c
	gcc $(FLAGS) -O0 $^ -o $@

matrixMult_O3: matrixMult.c
	gcc $(FLAGS) -O3 $^ -o $@

matrixMult_dyn_O0: matrixMult_dyn.c
	gcc $(FLAGS) -O0 $^ -o $@

matrixMult_dyn_O3: matrixMult_dyn.c
	gcc $(FLAGS) -O3 $^ -o $@


clean:
		rm -f $(OBJS)

