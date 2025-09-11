(deftemplate comportamiento_motor
   (slot estado))  ;; Definir estado

(deftemplate potencia
   (slot estado))  ;; Definir estado

(deftemplate combustible_en_motor
   (slot estado))  ;; Definir estado

(deftemplate indicador_bateria
   (slot valor))   ;; Definir valor

(deftemplate indicador_combustible
   (slot valor))   ;; Definir valor

(deftemplate inspeccion_fusible
   (slot resultado)) ;; Definir resultado

(deftemplate fusible
   (slot estado))  ;; Definir estado

(deftemplate bateria
   (slot nivel))   ;; Definir nivel

(deftemplate deposito_combustible
   (slot nivel))   ;; Definir nivel

(defrule motor_no_arranca
  (comportamiento_motor (estado no_arranca))
  =>
  (assert (potencia (estado desconectada))))

(defrule motor_se_para
  (comportamiento_motor (estado se_para))
  =>
  (assert (combustible_en_motor (estado falso))))
  
  (defrule potencia_sin_bateria
  (potencia (estado desconectada))
  (indicador_bateria (valor cero))
  =>
  (assert (bateria (nivel baja))))
  
  (defrule potencia_sin_fusible
  (potencia (estado desconectada))
  (inspeccion_fusible (resultado roto))
  =>
  (assert (fusible (estado fundido))))

(defrule sin_combustible
  (combustible_en_motor (estado falso))
  (indicador_combustible (valor cero))
  =>
  (assert (deposito_combustible (nivel vacio))))



