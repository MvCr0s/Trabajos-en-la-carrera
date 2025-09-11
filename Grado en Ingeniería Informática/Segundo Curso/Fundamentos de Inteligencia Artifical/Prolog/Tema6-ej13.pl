primeros([],_).
primeros([_|[]],[]):-!.
primeros([X|Cola],[X|L]):- primeros(Cola,L).
