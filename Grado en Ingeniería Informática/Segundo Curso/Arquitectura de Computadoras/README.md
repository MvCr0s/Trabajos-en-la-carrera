# 🖥️ Arquitectura de Computadores

## 📌 Introducción
Asignatura dedicada al estudio del **hardware a bajo nivel**, la organización interna del procesador y su relación con el rendimiento.  
En las practicas utilizábamos **lenguaje ensamblador MIPS** y resolución de problemas de segmentación y predicción de saltos.

---

## 🔹 Contenidos principales
- **Segmentación (Pipeline)**
  - Etapas en MIPS: IF, ID, EX, MEM, WB
  - Analogía con procesos industriales (lavandería)
  - Riesgos: estructurales, dependencias de datos y de control
  - Técnicas de anticipación (forwarding), inserción de *nop*, reordenación
  - Excepciones e interrupciones en pipeline

- **Predicción de saltos**
  - Riesgos de control y penalización por bifurcaciones
  - Estrategias:
    - Salto retardado (*delayed branching*)
    - Predicción estática (tomado/no tomado, backward-taken/forward-not-taken)
    - Predicción dinámica (1 bit, 2 bits)
    - Predictores correlacionados e híbridos (globales, locales, en modo torneo)
  - Implementaciones reales (ARM Cortex A53, Intel Core i7)

- **Lenguaje ensamblador MIPS**
  - Registros: `$t0-$t9`, `$s0-$s7`, `$a0-$a3`, `$v0-$v1`
  - Instrucciones básicas: `add`, `sub`, `lw`, `sw`, `beq`, `bne`, `j`
  - Uso de **syscalls** para entrada/salida:
    ```asm
    li $v0, 4       # imprimir string
    la $a0, msg
    syscall
    ```

---


## 🎯 Objetivos de la asignatura
- Comprender cómo se organiza y optimiza un procesador.
- Analizar el impacto de la segmentación y los riesgos en el rendimiento.
- Estudiar técnicas de predicción de saltos y su relevancia en procesadores modernos.
- Desarrollar y simular programas en ensamblador **MIPS**.

---
