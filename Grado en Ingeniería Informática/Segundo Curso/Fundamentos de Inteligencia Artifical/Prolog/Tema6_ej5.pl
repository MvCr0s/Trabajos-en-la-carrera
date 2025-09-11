insertar(_,[],[]).
insertar(Elem,Lista,[Elem|Lista]).
insertar(Elem,[Cabeza|Cola],[Cabeza|Cola1]):-insertar(Elem,Cola,Cola1).
