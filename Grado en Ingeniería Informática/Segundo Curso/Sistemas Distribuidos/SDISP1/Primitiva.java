public enum Primitiva {
    ADD,            // agrega un nodo hijo a un nodo padre SocketAddress+TSPadre  
    ADD_OK,         // ADD correcto 
    MULTICAST,      // difunde una variable desde un nodo nombreVariable+valor+TS  
    MULTICAST_OK,   // multicast iniciado
    TRANSMISSION,   // X: transmisión intermedia de un multicast nombreVariable+valor
    TRANSMISSION_OK,// transmisión iniciada
    RETURN,         // retorno de un HASHMAPS String con el valor de un HASHMAP 
    RETURN_END,     // end de los HASHMAPs
    NOTHING,        // mensaje vacío
    BAD_TS,         // error en la validación del token recibido
    NOTUNDERSTAND;  // (error protocolo)
}  
