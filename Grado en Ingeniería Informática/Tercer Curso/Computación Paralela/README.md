# ⚡ Computación Paralela

## 📌 Introducción
La asignatura estudia la **ejecución simultánea de operaciones** en múltiples unidades de procesamiento.  
Se analizan los **límites físicos y algorítmicos del paralelismo**, así como tres grandes modelos de programación: **OpenMP, MPI y CUDA**.

---

## 🔹 Contenidos principales

### 🧠 Fundamentos
- Clasificación de arquitecturas: SISD, SIMD, MISD, MIMD.  
- Modelos de memoria: compartida, distribuida e híbrida.  
- Paralelismo de **datos** y **tareas**.  
- Speedup, escalabilidad y leyes del paralelismo (Amdahl, Gustafson).  

---

### 🧵 OpenMP (memoria compartida)
- API para C, C++ y Fortran mediante directivas.  
- Regiones paralelas, bucles paralelizados y secciones independientes.  
- Control de hilos y planificación del trabajo.  
- Variables compartidas y privadas.  
- Mecanismos de reducción.  
- Estrategias de planificación y sincronización.  

---

### 🌍 MPI (memoria distribuida)
- Biblioteca estándar para **paso de mensajes** en sistemas distribuidos.  
- Cada proceso usa su propia memoria local y se comunica mediante mensajes.  
- Inicialización y finalización del entorno MPI.  
- Identificación y organización de procesos mediante grupos y comunicadores.  
- Envío, recepción y sincronización de mensajes.  
- Comunicaciones colectivas (broadcast, gather, scatter, reduce).  
- Modelo de ejecución **SPMD (Single Program, Multiple Data)**.  

---

### 🎮 CUDA (GPU – GPGPU)
- Plataforma de **NVIDIA** para cómputo paralelo en GPUs.  
- Modelo jerárquico de ejecución: grids, bloques e hilos.  
- Identificadores de hilos y bloques para organizar el paralelismo.  
- Jerarquía de memoria: global, compartida, registros y cachés.  
- Sincronización de hilos dentro de un bloque.  
- Gestión de memoria y transferencia entre CPU y GPU.  
- Optimización de kernels para maximizar el rendimiento.  

---

## 🎯 Objetivos de la asignatura
- Comprender los fundamentos del paralelismo y sus limitaciones.  
- Conocer los principales modelos de programación paralela.  
- Aprender a diseñar algoritmos paralelos eficientes.  
- Aplicar técnicas con OpenMP, MPI y CUDA en la práctica.  
