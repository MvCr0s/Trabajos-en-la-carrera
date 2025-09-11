borrar(_,[],[]).
borrar(Elem,[Elem|Cola],Cola).
borrar(Elem,[Cabeza|Cola],[Cabeza|Cola1]):-borrar(Elem,Cola,Cola1).
