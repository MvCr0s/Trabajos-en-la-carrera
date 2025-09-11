(deftemplate oav-u "Plantilla Hechos univaluados"
	(slot objeto (type SYMBOL))
	(slot atributo(type SYMBOL))
	(slot valor)
)
(deftemplate oav-m "Plantilla Hechos multivaluados"
	(slot objeto (type SYMBOL))
	(slot atributo(type SYMBOL))
	(slot valor)
)

(defrule garantizar-univaluados
    (declare (salience 10000))
    ?x <- (oav-u (objeto ?o1) (atributo ?a1) (valor ?v1))
    ?y <- (oav-u (objeto ?o1) (atributo ?a1) (valor ?v2))
    (test (> (fact-index ?x) (fact-index ?y)))
=>
(retract ?y)) 



;
; Reglas para determinar el escenario


(defrule Solicitar-escenario "solicitamos el dispositivo del que se quiere hacer un diagnóstico"
=> 
	(printout T crlf crlf "Bienvenido al asistente para el diagnostico de dispositivos electronicos." crlf)
	(printout T "Dispone de tres dispositivos para diagnosticar." crlf "Responde 1 para el primer escenario-movil, 2 para el segundo-tablet y 3 para el tercero-portatil:")
	(assert (escenario =(read T))) ; el usuario introduce el valor por entrada estandar
	(printout T crlf)
)

(defrule Comprobar-escenario  "comprobamos que el escenario introducido es correcto"
	(escenario ?x) 
	(not (escenario 1))
	(not (escenario 2))
	(not (escenario 3))
=>
	(reset)  ; si no es correcto, hacemos un reset
	(run)  ; y ejecutamos de nuevo el programa
)

(defrule Activar-escenario-1  "si el dispositivo es un móvil, introducimos en la memoria de trabajo los hechos iniciales correspondientes"
	(escenario 1) 

=>
	(assert (oav-u (objeto movil) (atributo antiguedad) (valor 14)))
	(assert (oav-u (objeto movil) (atributo sistema-operativo) (valor actual)))
	(assert (oav-m (objeto movil) (atributo fallo-presente) (valor se-apaga-inesperadamente)))
)


(defrule Activar-escenario-2  "si el dispositivo es una tablet, introducimos en la memoria de trabajo los hechos iniciales correspondientes"
	(escenario 2) 

=>
	(assert (oav-u (objeto tablet) (atributo antiguedad) (valor 20)))
	(assert (oav-u (objeto tablet) (atributo sistema-operativo) (valor actual)))
	(assert (oav-m (objeto tablet) (atributo fallo-presente) (valor error-arranque)))
	(assert (oav-m (objeto tablet) (atributo fallo-presente) (valor ficheros-corruptos)))
	(assert (oav-m (objeto tablet) (atributo fallo-presente) (valor en-bateria)))
	(assert (oav-m (objeto tablet) (atributo fallo-presente) (valor se-apaga-inesperadamente)))
	(assert (oav-m (objeto tablet) (atributo fallo-presente) (valor golpes)))
)

(defrule Activar-escenario-3  "si el dispositivo es un portátil, introducimos en la memoria de trabajo los hechos iniciales correspondientes"
	(escenario 3) 

=>
	(assert (oav-u (objeto portatil) (atributo antiguedad) (valor 23)))
	(assert (oav-u (objeto portatil) (atributo sistema-operativo) (valor noactual)))
	(assert (oav-m (objeto portatil) (atributo fallo-presente) (valor no-enciende)))
)

(defrule en-revision "si esta en garantia y el SO es actual, el dispositivo pasa a revision"
		(oav-u (objeto ?x) (atributo antiguedad) (valor ?v &:(<= ?v 24)))
		(oav-u  (objeto ?x) (atributo sistema-operativo) (valor actual))
=>
        (assert (oav-u (objeto ?x) (atributo estado) (valor en-revision)))

)  

(defrule rechazado "si no esta en garantía o el SO no es actual, el dispositivo es rechazado"
  		(oav-u (objeto ?x)(atributo sistema-operativo))  ; Se utiliza esta sentencia para que la regla no dé error en caso de cumplirse solo el segundo patron del or. 
														 ; Así podemos usar la variable ?x en el lado derecho de la regla.
														 ; Además, debemos utilizar un atributo univaluado para que la regla no se active varias veces.	
		(or (oav-u (objeto ?x) (atributo antiguedad) (valor ?v &:(> ?v 24)))
		(not (oav-u  (objeto ?x) (atributo sistema-operativo) (valor actual))))
=>
        (assert (oav-u (objeto ?x) (atributo estado) (valor rechazado)))

)  

; Si el dispositivo pasa a revision, se diagnostica la posible causa mediante las siguientes reglas


(defrule falloalimentacion  "esta regla determina cuándo la causa es un fallo de alimentación"
		(declare (salience 1000)) ; La prioridad es más alta que la de las demas reglas excepto la de garantizar univaluados, que debe tener la más alta del programa.
								  ; De esta forma, esta regla se ejecuta antes que las demás reglas de diagnóstico, ya que la regla que determina si el fallo es de sistema operativo
								  ; necesita saber si hay un fallo de alimentación para que el not funcione adecuadamente. Si esta regla no se ejecuta antes, la de fallo de sistema operativo 
								  ; podría activarse y ejecutarse incluso habiendo fallo de alimentación dependiendo de la estrategia de resolución de conflictos que se use, 
								  ; dando un diagnóstico erróneo.
		(oav-u (objeto ?x) 		  
                (atributo estado) 
                (valor en-revision))
				
        (oav-m  (objeto ?x) 
                (atributo fallo-presente) 
                (valor en-bateria))
(or     (oav-m  (objeto ?x) 
                (atributo fallo-presente) 
                (valor se-apaga-inesperadamente))
        (oav-m  (objeto ?x) 
                (atributo fallo-presente) 
                (valor no-enciende)))

=>
        (assert (oav-m (objeto ?x) 
                (atributo diagnostico) 
                (valor fallo-alimentacion) 
				))
)  


(defrule error-sistema-operativo   "esta regla determina cuándo la causa es error en el sistema operativo"

        (oav-u (objeto ?x) 
                (atributo estado) 
                (valor en-revision))
		(not (oav-m (objeto ?x) 
                (atributo diagnostico) 
                (valor fallo-alimentacion) 
				)
		)  ; Se ha tenido que haber diagnosticado antes fallo de alimentación
        (or (oav-m (objeto ?x) 
                (atributo fallo-presente) 
                (valor error-arranque))
		(oav-m  (objeto ?x) 
                (atributo fallo-presente) 
                (valor ficheros-corruptos)))

=>
        (assert (oav-m (objeto ?x) 
                (atributo diagnostico) 
                (valor error-sistema-operativo) 
				))
)  





(defrule mal-uso  "esta regla determina cuándo la causa es un mal uso"

       (oav-m (objeto ?x) 
                (atributo diagnostico) 
                (valor fallo-alimentacion) 
				)
        (oav-m  (objeto ?x) 
                (atributo fallo-presente) 
                (valor golpes))

=>
        (assert (oav-m (objeto ?x) 
                (atributo diagnostico) 
                (valor mal-uso) 
				))
)  


; las reglas que informan al usuario tienen una prioridad minima en el programa


(defrule informar1  "si el dispositivo tiene diagnostico, se informa"
	(declare (salience -1000)) ; Se disminuye la prioridad a la mínima del programa tanto en esta regla como en las dos siguientes 
							   ; para asegurar que todas las reglas de diagnóstico que se pudieran activar y ejecutar ya lo han hecho
	(oav-m (objeto ?x)
		(atributo diagnostico) 
		(valor ?v))
	=>
	(printout t "El dispositivo " ?x " presenta el fallo " ?v crlf)
)

(defrule informar2  "si el dispositivo no ha sido rechazado, y no se ha llegado a un diagnostico, se informa"
	(declare (salience -1000))   
	(oav-u (objeto ?x)(atributo sistema-operativo)) ; Es necesario para poder usar la variable ?x en el lado derecho de la  regla. 
													; Usamos para ello un atributo univaluado para evitar que se active la regla varias veces para el mismo dispositivo.
	(not(oav-m(objeto ?x)
		(atributo diagnostico) 
		(valor ?valor)))
	(not (oav-u (objeto ?x)
		(atributo estado) 
		(valor rechazado)))

	=>
	(printout t "El dispositivo " ?x " no puede ser diagnosticado."crlf)
)


(defrule informar3  "si el dispositivo ha sido rechazado, se informa"
	(declare (salience -1000))
	(oav-u (objeto ?x)
		(atributo estado) 
		(valor rechazado))
	=>
	(printout t "El dispositivo " ?x " ha sido rechazado por no estar en garantia o no tener un sistema operativo actual."crlf)
)



