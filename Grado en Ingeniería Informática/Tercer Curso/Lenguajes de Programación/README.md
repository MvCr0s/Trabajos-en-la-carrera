# 📝 Lenguajes de Programación

## 📌 Introducción
Asignatura dedicada al **estudio de los lenguajes de programación**: su diseño, implementación y uso.  
Se analizan los **paradigmas de programación**, la **evolución histórica**, así como las técnicas de **procesamiento de lenguajes** (compiladores e intérpretes).

---

## 🔹 Contenidos principales

### 🧠 Fundamentos:
- Origen y evolución de los lenguajes de programación.  
- Generaciones: máquina, ensamblador, alto nivel, específicos, inteligencia artificial.  
- Características de los LdP: abstracción, expresividad, eficiencia, reusabilidad.  
- Paradigmas: imperativo, funcional, lógico, orientado a objetos.  
- Implementación: compiladores, intérpretes y soluciones híbridas.  
- Fases de compilación: análisis léxico, sintáctico, semántico, generación y optimización de código.

---

### 🔤 Análisis léxico:
- Función: transformar caracteres → **tokens**.  
- Creación de la tabla de símbolos.  
- Eliminación de comentarios y macros.  
- Expresiones regulares para definir lenguajes.  
- Autómatas finitos deterministas (AFD) y no deterministas (AFN).  
- Generadores automáticos de analizadores léxicos (LEX).

---

### 📐 Sintaxis y gramáticas:
- **Jerarquía de Chomsky**: lenguajes regulares, independientes de contexto, dependientes de contexto y no restringidos.  
- Gramáticas independientes de contexto: reglas, derivaciones, árboles sintácticos.  
- Ambigüedad y recursividad (izquierda/derecha).  
- Eliminación de símbolos inútiles, reglas ε y recursión.  
- Factorización por la izquierda para evitar ambigüedades.

---

### 🔽 Análisis sintáctico descendente:
- Estrategia **top-down** (predictiva).  
- Uso de los conjuntos **FIRST** y **FOLLOW**.  
- Construcción de la **tabla de análisis predictivo (TASP)**.  
- Gramáticas LL(1): sin ambigüedad, sin recursión izquierda y factorizadas.  
- Técnicas de recuperación de errores.

---

### 🔼 Análisis sintáctico ascendente:
- Estrategia **bottom-up** (shift-reduce).  
- Uso de pila para reducir a reglas gramaticales.  
- Métodos LR(0), SLR(1), LALR(1).  
- Conflictos de **desplazamiento/reducción** y **reducción/reducción**.  
- Construcción de tablas LR y su aplicación en compiladores.  

---


## 🎯 Objetivos de la asignatura
- Conocer los **fundamentos y evolución** de los lenguajes de programación.  
- Entender y aplicar los **paradigmas de programación**.  
- Comprender las fases de un **compilador** y el procesamiento de programas.  
- Definir lenguajes mediante **expresiones regulares y gramáticas**.  
- Construir y analizar **autómatas, parsers descendentes y ascendentes**.  
- Manejar herramientas para la generación automática de compiladores.  
