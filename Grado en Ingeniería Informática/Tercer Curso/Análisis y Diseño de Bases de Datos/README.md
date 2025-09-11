# 🗄️ Análisis y Diseño de Bases de Datos (ADBD)

## 📌 Introducción
Asignatura dedicada al **diseño, implementación y gestión de bases de datos**, desde el modelado conceptual hasta su explotación con lenguajes de consulta y técnicas avanzadas.  
Se estudian los **Sistemas Gestores de Bases de Datos (SGBD)**, la normalización, SQL, así como enfoques modernos como **NoSQL y Big Data**.

---

## 🔹 Contenidos principales

### 📖 Fundamentos:
- Datos, información y conocimiento.  
- Problemas del almacenamiento en ficheros.  
- SGBD: funciones principales (almacenamiento, concurrencia, seguridad, recuperación).  
- Niveles de abstracción: conceptual, lógico, físico.  
- Independencia lógico-física.  

### 📝 Modelado de datos:
- Análisis de requisitos.  
- Diseño conceptual → **Modelo Entidad-Relación (ER)**.  
- Atributos, claves primarias y alternativas.  
- Cardinalidad y roles, relaciones N:M, entidades débiles.  
- Generalización/especialización e ISA.  
- Restricciones y trampas de modelado.  

### 📊 Modelo relacional:
- Origen (Codd, 1970) y ventajas.  
- Tablas, atributos, tuplas e instancias.  
- Claves primarias, candidatas y foráneas.  
- Integridad referencial y restricciones.  
- Traducción ER → Relacional.  
- Representación de relaciones 1:1, 1:N, N:M.  
- Jerarquías ISA en el modelo relacional.  
- Vistas y seguridad.  

### 💻 SQL:
- **DDL** (definición): `CREATE`, `ALTER`, `DROP`.  
- **DML** (manipulación): `INSERT`, `UPDATE`, `DELETE`.  
- **DQL** (consulta): `SELECT` con filtros, joins y subconsultas.  
- Operadores: `UNION`, `INTERSECT`, `EXCEPT`.  
- Funciones de agregación: `COUNT`, `AVG`, `SUM`, `MAX`, `MIN`.  
- `GROUP BY` y `HAVING`.  
- Restricciones (`CHECK`, `ASSERTION`) y **triggers**.  

### 🧮 Normalización:
- Problema de la redundancia.  
- Dependencias funcionales (DF).  
- Formas normales:  
  - 1NF (eliminar grupos repetitivos).  
  - 2NF (atributos dependientes de clave completa).  
  - 3NF (evitar dependencias transitivas).  
  - BCNF (solo dependencias con la clave).  
  - 4NF y 5NF (dependencias multivaluadas y de unión).  
- Algoritmos de descomposición.  

### ⚙️ Administración de BD:
- Roles: usuarios, desarrolladores y administradores (DBA).  
- Instalación, arranque y parada de BD.  
- Creación y gestión de usuarios y privilegios.  
- Optimización de consultas y rendimiento (tuning).  
- Diccionario de datos y metadatos en PostgreSQL.  

---

## 🎯 Objetivos de la asignatura
- Comprender el papel de los **SGBD** en la gestión de la información.  
- Diseñar bases de datos desde el nivel conceptual hasta el físico.  
- Dominar el uso de **SQL** y garantizar la integridad de los datos.  
- Aplicar **normalización** para evitar redundancias y anomalías.  
- Administrar un sistema de bases de datos de forma eficiente.  
- Conocer tendencias modernas: **NoSQL, Big Data y Web Semántica**.  
