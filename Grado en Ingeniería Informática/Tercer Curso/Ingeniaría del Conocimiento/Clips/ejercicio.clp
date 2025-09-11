(deftemplate oav-u
	(slot objeto (type SYMBOL))
	(slot atributo(type SYMBOL))
	(slot valor)
)

(deftemplate oav-m
	(slot objeto (type SYMBOL))
	(slot atributo(type SYMBOL))
	(slot valor)
)
(defrule semanticaUnivaluadaOAV
	?x<-(oav-u  (objeto ?objeto)
		(atributo ?atributo)
		(valor ?valor1))
	?y<-(oav-u  (objeto ?objeto)
		(atributo ?atributo)
		(valor ?valor2 & ~?valor1))
	(test(< (fact-index ?x) (fact-index ?y)))
	=>
	(retract ?x)
)


(deffacts hechosInicialesMovil
	(oav-u (objeto movil) (atributo nombre) (valor movil))
	(oav-u (objeto movil) (atributo antiguedad) (valor 14))
	(oav-u (objeto movil) (atributo SO) (valor actual)) ; Cambiado para que coincida con "rechazado"
	(oav-m (objeto movil) (atributo fallo) (valor apagaInesperadamente))
)
(deffacts hechosInicialesTablet
	(oav-u (objeto tablet) (atributo nombre) (valor tablet))
	(oav-u (objeto tablet) (atributo antiguedad) (valor 20))
	(oav-u (objeto tablet) (atributo SO) (valor actual))
	(oav-m (objeto tablet) (atributo fallo) (valor errorArranque))
	(oav-m (objeto tablet) (atributo fallo) (valor ficheroCorruptos))
	(oav-m (objeto tablet) (atributo fallo) (valor apagaInesperadamente))
	(oav-m (objeto tablet) (atributo fallo) (valor falloBateria))
	(oav-m (objeto tablet) (atributo fallo) (valor golpes))
)
(deffacts hechosInicialesPortatil
	(oav-u (objeto portatil) (atributo nombre) (valor portatil))
	(oav-u (objeto portatil) (atributo antiguedad) (valor 23))
	(oav-u (objeto portatil) (atributo SO) (valor noActual))
	(oav-m (objeto portatil) (atributo fallo) (valor noEnciende))
)



(defrule garantia
	(oav-u (objeto ?x) (atributo antiguedad) (valor ?y))
	(test (<= ?y 24)) ; Menor o igual a 24 meses
	(oav-u (objeto ?x) (atributo SO) (valor actual))
	=>
	(assert (oav-u (objeto ?x)(atributo garantia) (valor true)))
	
)

(defrule rechazado
  	(oav-u (objeto ?x)(atributo SO)) 	
	(or (oav-u (objeto ?x) (atributo antiguedad) (valor ?v &:(> ?v 24)))
	(not (oav-u  (objeto ?x) (atributo SO) (valor actual))))
=>
        (assert (oav-u (objeto ?x) (atributo garantia) (valor false)))

) 



(defrule sistemaAlimentacion-fallo
	(declare(salience 1000))
	(oav-u (objeto ?x)
		(atributo garantia)
		(valor true))
	(oav-m (objeto ?x)
		(atributo fallo)
		(valor falloBateria))
	(or (oav-m (objeto ?x)
		(atributo fallo)
		(valor apagaInesperadamente))
	    (oav-m (objeto ?x)
		(atributo fallo)
		(valor noEnciende)))
	=>
	(assert (oav-m (objeto ?x)
		(atributo fallos)
		(valor sistemaAlimentacion)))
)


(defrule sistemaOperativo-fallo
	(oav-u  (objeto ?x)
		(atributo garantia)
		(valor true))
	(not (oav-m (objeto ?x)
		(atributo fallo)
		(valor sistemaAlimentacion)))
	(or (oav-m (objeto ?x)
		(atributo fallo)
		(valor errorArranque))
	    (oav-u (objeto ?x)
		(atributo fallo)
		(valor ficherosCorruptos)))
	=>
	(assert (oav-m (objeto ?x)
		(atributo fallos)
		(valor sistemaOperativo)))
)



(defrule malUso-fallo
	(oav-u  (objeto ?x)
		(atributo garantia)
		(valor true))
	(oav-m  (objeto ?x)
		(atributo fallos)
		(valor sistemaAlimentacion))
	(oav-m  (objeto ?x)
		(atributo fallo)
		(valor golpes))
	=>
	(assert (oav-m (objeto ?x)
		(atributo fallos)
		(valor malUso)))
)




(defrule resultados
	(declare (salience -1000)) 
	(oav-u (objeto ?x) 
	       (atributo nombre) 
	       (valor ?z)) ; Obtener el nombre del dispositivo (movil, tablet, portatil)
	(oav-m (objeto ?x) 
	       (atributo fallos) 
	       (valor ?y)) ; Obtener el tipo de fallo (sistema operativo, mal uso, etc.)
	=>
	(printout t "El dispositivo " ?x ", llamado " ?z ", presenta el fallo " ?y crlf)
)



(defrule resultados2
	(declare (salience -1000)) 
	(oav-u  (objeto ?x)
		(atributo garantia)
		(valor false))
	(oav-u (objeto ?x) 
	       (atributo nombre) 
	       (valor ?z)) ; Obtener el nombre del dispositivo (movil, tablet, portatil)
		=>
	(printout t "El dispositivo " ?x ", llamado " ?z ",no tiene garantia " crlf)
)



(defrule resultados3
	(declare (salience -1000)) 
	(oav-u  (objeto ?x)
		(atributo garantia)
		(valor true))
	(oav-u (objeto ?x) 
	       (atributo nombre) 
	       (valor ?z)) ; Obtener el nombre del dispositivo (movil, tablet, portatil)
	(not(oav-m (objeto ?x) 
	       (atributo fallos) 
	       (valor ?y))) 
		=>
	(printout t "El dispositivo " ?x ", llamado " ?z " no se ha encontrado cual es el fallo" crlf)
)


	




