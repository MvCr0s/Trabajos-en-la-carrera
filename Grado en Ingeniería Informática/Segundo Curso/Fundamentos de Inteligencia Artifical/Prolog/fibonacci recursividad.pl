f(0,1).
f(1,1).
f(N,R):- N>1, N1 is N-1, N2 is N-2, f(N1,R1), f(N2,R2), R is R1+R2.
