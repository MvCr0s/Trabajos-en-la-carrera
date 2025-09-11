
(deftemplate oav-u
	(slot objeto (type SYMBOL))
	(slot atributo(type SYMBOL))
	(slot valor)
)

(deftemplate oav-m
	(slot objeto (type SYMBOL))
	(slot atributo (type SYMBOL))
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
(deffacts hechos-iniciales
	(oav-u (objeto outside)
		(atributo live)
		(valor true))
	(oav-m (objeto w5)
		(atributo conectado)
		(valor outside))
	(oav-m (objeto l1)
		(atributo conectado)
		(valor w0))
	(oav-m (objeto l2)
		(atributo conectado)
		(valor w4))
)
(deffacts hechos-iniciales-bombillas
	(oav-u (objeto l2)
		(atributo light)
		(valor true))
	(oav-u (objeto l1)
		(atributo light)
		(valor true))
	(oav-u (objeto l1)
		(atributo ok)
		(valor true))
	(oav-u (objeto l2)
		(atributo ok)
		(valor true))
		
)
(deffacts hechos-iniciales-cb
	(oav-u (objeto cb1)
		(atributo ok)
		(valor true))
	(oav-u (objeto cb2)
		(atributo ok)
		(valor true))
		
)
(deffacts hechos-iniciales-switch
	(oav-u (objeto s3)
		(atributo estado)
		(valor up))
	(oav-u (objeto s2)
		(atributo estado)
		(valor up))
	(oav-u (objeto s1)
		(atributo estado)
		(valor down))
	(oav-u (objeto s3)
		(atributo ok)
		(valor true))
	(oav-u (objeto s2)
		(atributo ok)
		(valor true))
	(oav-u (objeto s1)
		(atributo ok)
		(valor true))
		
)

(defrule bombillaLit 
	(oav-u (objeto ?x)
		(atributo light)
		(valor true))
	(oav-u (objeto ?x)
		(atributo live)
		(valor true))
	(oav-u (objeto ?x)
		(atributo ok)
		(valor true))
	=>
	(assert(oav-u (objeto ?x)
		(atributo lit)
		(valor true)))
)

(defrule tension
	(oav-m (objeto ?x)
		(atributo conectado)
		(valor ?y))
	(oav-u (objeto ?y)
		(atributo lit)
		(valor true))
	=>
	(assert(oav-u (objeto ?x)
		(atributo lit)
		(valor true)))	
)

(defrule conectados1
	(oav-u (objeto cb2)
		(atributo ok)
		(valor true))
		
	=>
	(assert(oav-m (objeto w6)
		(atributo conectado)
		(valor w5)))
)

(defrule conectados2
	(oav-u (objeto cb1)
		(atributo ok)
		(valor true))
		
	=>
	(assert(oav-m (objeto w3)
		(atributo conectado)
		(valor w5)))
)

(defrule conectados3
	(oav-u (objeto s2)
		(atributo estado)
		(valor up))
	(oav-u (objeto s2)
		(atributo ok)
		(valor true))		
		
	=>
	(assert(oav-m (objeto w0)
		(atributo conectado)
		(valor w1)))
)

(defrule conectados4
	(oav-u (objeto s2)
		(atributo estado)
		(valor down))
	(oav-u (objeto s2)
		(atributo ok)
		(valor true))		
		
	=>
	(assert(oav-m (objeto w0)
		(atributo conectado)
		(valor w2)))
)

(defrule conectados5
	(oav-u (objeto s1)
		(atributo estado)
		(valor down))
	(oav-u (objeto s1)
		(atributo ok)
		(valor true))		
		
	=>
	(assert(oav-m (objeto w5)
		(atributo conectado)
		(valor w2)))
)

(defrule conectados5
	(oav-u (objeto s1)
		(atributo estado)
		(valor up))
	(oav-u (objeto s1)
		(atributo ok)
		(valor true))		
		
	=>
	(assert(oav-m (objeto w5)
		(atributo conectado)
		(valor w1)))
)

(defrule conectados5
	(oav-u (objeto s3)
		(atributo estado)
		(valor up))
	(oav-u (objeto s3)
		(atributo ok)
		(valor true))		
		
	=>
	(assert(oav-m (objeto w4)
		(atributo conectado)
		(valor w3)))
)
