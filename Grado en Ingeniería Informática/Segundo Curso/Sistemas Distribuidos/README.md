# 🌐 Sistemas Distribuidos

## 📌 Introducción
Asignatura centrada en el estudio de los **sistemas distribuidos**, donde varios procesos o nodos conectados en red cooperan mediante **comunicación por paso de mensajes**.  
Se abordaron aspectos teóricos, arquitecturas, middleware y prácticas en **Java (sockets, RMI, CORBA)**.

---

## 🔹 Contenidos principales

### 🏗️ Conceptos básicos:
- Definición: sistemas donde los componentes se comunican sólo mediante **paso de mensajes**.
- Modelos: minicomputadoras, workstations, cluster, grid computing.
- Ventajas: escalabilidad, fiabilidad, compartición de recursos, trabajo cooperativo.
- Desafíos: heterogeneidad, concurrencia, seguridad, tolerancia a fallos, transparencia.

### 🔌 Comunicación con sockets:
- **Sockets UDP**: no orientados a conexión, rápidos pero sin garantías de entrega ni orden.
- **Sockets TCP**: orientados a conexión, ofrecen secuenciamiento y fiabilidad.
- **NIO (Non-blocking I/O)**: operaciones asíncronas.
- **Sockets seguros (SSL/JSSE)**: confidencialidad y autenticación en Java.

### ✉️ Paso de mensajes:
- **IPC distribuido**: comunicación y sincronización.
- Comunicación síncrona vs. asíncrona.
- Operaciones bloqueantes y no bloqueantes.
- Semánticas de envío/recepción: síncrono-síncrono, asíncrono-síncrono, etc.
- **Modelo de actores**: procesos que intercambian mensajes asincrónicamente (Erlang, Akka).

### 📦 Representación de datos:
- Problema: la red transmite solo bytes → diferencias entre arquitecturas (endianness, Unicode).
- **Marshalling / Unmarshalling** (serialización y deserialización).
- Estándares: ASN.1, CORBA CDR, Java Serialization, JSON, XML.
- Middleware: CORBA IDL, RPC, Protocol Buffers.

### 🔗 Invocación remota de métodos:
- **Java RMI (Remote Method Invocation)**:
  - Objetos remotos, stubs, ROID, callbacks.
  - Patrones: Proxy, Factory, Observer.
  - Objetos activables (on-demand).
- **CORBA (Common Object Request Broker Architecture)**:
  - Interfaces en IDL, interoperabilidad multilenguaje.
  - ORB como intermediario entre cliente y servidor.
  - Servicios adicionales: nombres, transacciones, seguridad, eventos.

---


## 🎯 Objetivos de la asignatura
- Comprender la arquitectura y desafíos de los sistemas distribuidos.  
- Programar aplicaciones distribuidas en **Java** usando sockets, RMI y CORBA.  
- Analizar y aplicar técnicas de **sincronización y concurrencia** en entornos distribuidos.  
- Manejar problemas de **representación de datos, tolerancia a fallos y transparencia**.  
- Conocer modelos arquitectónicos cliente-servidor y orientados a mensajería.  
