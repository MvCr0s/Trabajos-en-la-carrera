# 🖥️ Fundamentos de Sistemas Operativos

## 📌 Introducción
Asignatura dedicada a los **principios básicos de los sistemas operativos**, su organización y los mecanismos fundamentales para la gestión de recursos.  
Incluye una parte práctica con **programación en C** y uso de **comandos UNIX**.

---

## 🔹 Contenidos principales

### 📖 Conceptos básicos:contentReference[oaicite:9]{index=9}
- Definición de sistema operativo: interfaz entre hardware y usuario.
- Recursos gestionados: CPU, memoria, ficheros, dispositivos de E/S.
- Servicios: sincronización, seguridad, comunicación, protección.
- Evolución: sistemas por lotes, multiprogramados y de tiempo compartido.
- Interrupciones, trampas y modo dual (usuario/núcleo).

### ⚙️ Lenguaje C (prácticas):contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
- Tipos de datos, variables, operadores, estructuras de control.
- Arrays, `struct`, punteros y funciones.
- Cadenas de caracteres y librerías estándar.
- Gestión dinámica de memoria (`malloc`, `calloc`, `realloc`, `free`).
- Ficheros: apertura, lectura/escritura, cierre.
- Descriptores de archivo y llamadas al sistema (`open`, `read`, `write`, `close`):contentReference[oaicite:12]{index=12}.

### 👥 Procesos e hilos:contentReference[oaicite:13]{index=13}:contentReference[oaicite:14]{index=14}
- Creación de procesos (`fork`, `exec`).
- Estados de procesos y planificación.
- Procesos pesados vs. hilos ligeros.
- Hilos con **pthreads** (`pthread_create`, `pthread_join`, `pthread_exit`).
- Sincronización con **semáforos POSIX** (`sem_wait`, `sem_post`).

### 📅 Planificación de CPU:contentReference[oaicite:15]{index=15}
- Algoritmos: FCFS, SJF, SRTF, Round Robin, prioridad.
- Planificación multinivel y multicolas con realimentación.
- Planificación en sistemas multiprocesador.
- Planificación en tiempo real: EDF, RMS.

### 🔒 Concurrencia y bloqueos:contentReference[oaicite:16]{index=16}
- Condiciones para el bloqueo mutuo (*deadlock*).
- Algoritmo del banquero.
- Detección y prevención de interbloqueos.

### 💻 Comandos UNIX:contentReference[oaicite:17]{index=17}
- Entrada/salida estándar, redirección y tuberías.
- Comandos avanzados: `grep`, `sort`, `uniq`, `wc`, `cut`, `tr`, `paste`.
- Uso de `scp` para transferir ficheros.
- Prácticas con scripting básico en shell.

---


## 🎯 Objetivos de la asignatura
- Comprender la estructura básica de un sistema operativo.
- Aprender a programar en **C** y entender llamadas al sistema.
- Manejar conceptos de **procesos, hilos y sincronización**.
- Conocer algoritmos de planificación y gestión de memoria.
- Usar comandos **UNIX/Linux** para administración básica.
