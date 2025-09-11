# 📈 Evaluación y Rendimiento de Sistemas Software (ERSS)

## 📌 Introducción
Asignatura centrada en el **análisis del rendimiento de sistemas informáticos**, aplicando métricas, modelos de colas y técnicas de evaluación.  
Se estudian métodos para medir y predecir el comportamiento de sistemas software y hardware bajo distintas cargas de trabajo.

---

## 🔹 Contenidos principales

### ⚙️ Métricas de rendimiento:
- **Tasa de llegada (λ)** y **tasa de servicio (μ)**.  
- **Utilización (ρ)** de los recursos.  
- **Tiempo de servicio (S)** y **tiempo de respuesta (R)**.  
- **Tiempo de espera en cola (W)**.  
- **Throughput (X)** y productividad.  
- **Ley de Little**: Q = λR.  
- **Ley de la Utilización** y **Ley del Flujo Forzado**.

### 🔄 Modelos de colas:
- **Redes abiertas**: llegadas y salidas externas, población variable.  
- **Redes cerradas**: número fijo de clientes, población constante.  
- Modelos M/M/1, M/M/m y variantes.  
- Condiciones de estabilidad: λ ≤ μ.  
- Aproximación de Schweitzer para redes cerradas.  
- Parámetros de rendimiento: Q, R, W, U en sistemas con múltiples servidores.

### 📝 Evaluación práctica:
- Ejercicios con **diagramas de secuencia** y **actividad** para modelar restricciones temporales.  
- Casos de estudio en **sistemas de control** (ascensores, bancos, molinos de viento).  
- Procesamiento de pedidos en e-commerce con concurrencia y sincronización.  
- Análisis de sistemas paralelos con **MPI, OpenMP y CUDA** representados como diagramas de actividad.

### 🔬 Benchmarking y comparación de sistemas:
- Métricas clásicas: **MIPS, MFLOPS, CPI, throughput, latencia**.  
- Comparación de procesadores mediante **media aritmética, geométrica y armónica**.  
- Benchmarks para servidores web:  
  - Throughput vs. latencia.  
  - Escalabilidad horizontal con servidores adicionales.  
  - Impacto de la latencia de red.  
- Métricas de equidad: **Coeficiente de Variación (CV)** e **Índice de Jain (JFI)**.  
- Disponibilidad en clústeres de servidores.  

---

## 🎯 Objetivos de la asignatura
- Comprender y aplicar **métricas de rendimiento** en sistemas software y hardware.  
- Modelar sistemas mediante **teoría de colas** y aplicar la **Ley de Little**.  
- Analizar el rendimiento en términos de **utilización, throughput y latencia**.  
- Comparar alternativas mediante **benchmarks y métricas objetivas**.  
- Detectar cuellos de botella y proponer mejoras de rendimiento.  
