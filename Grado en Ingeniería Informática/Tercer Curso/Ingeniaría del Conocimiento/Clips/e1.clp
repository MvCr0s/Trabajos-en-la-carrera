(deftemplate oav-u
	(slot objeto (type SYMBOL))
	(slot atributo(type SYMBOL))
	(slot valor)
)


(deffacts hechos-no-univaluados
  (oav-u (objeto Juan) (atributo edad) (valor 35))
  (oav-u (objeto Juan) (atributo edad) (valor 35))
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
