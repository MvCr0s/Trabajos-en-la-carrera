# 🏗️ Diseño de Software

## 📌 Introducción
Asignatura dedicada a los **principios y patrones de diseño de software orientado a objetos**.  
Se centra en el uso de **patrones GRASP** y **patrones de diseño (GoF)** para mejorar la calidad, mantenibilidad y reutilización del código.

---

## 🔹 Contenidos principales

### 🎯 Patrones GRASP:contentReference[oaicite:10]{index=10}
- **Experto** → asignar responsabilidades a la clase que tiene la información necesaria.  
- **Creador** → una clase crea instancias de otra cuando las contiene, las usa o tiene datos de inicialización.  
- **Bajo acoplamiento** → reducir dependencias entre clases.  
- **Alta cohesión** → responsabilidades muy relacionadas en cada clase.  
- **Controlador** → clase que gestiona eventos del sistema.  
- **Fabricación pura** → asignar responsabilidades a una clase auxiliar para no romper cohesión.  
- **Polimorfismo** → evitar condicionales basados en tipo.  
- **Indirección** → usar intermediarios para reducir acoplamiento.  
- **Variaciones protegidas** → encapsular puntos de cambio para proteger el sistema frente a modificaciones.  

---

### 📖 Patrones de diseño (GoF)

#### 🧩 Creacionales
- **Singleton**:contentReference[oaicite:11]{index=11} → garantiza una única instancia y acceso global.  
- **Factory Method**:contentReference[oaicite:12]{index=12} → subclases deciden qué objeto concreto instanciar.  
- **Abstract Factory**:contentReference[oaicite:13]{index=13} → crear familias de objetos relacionados sin conocer las clases concretas.  

#### 🏗️ Estructurales
- **Adaptador (Adapter)**:contentReference[oaicite:14]{index=14} → convierte la interfaz de una clase en otra esperada por el cliente.  
- **Compuesto (Composite)**:contentReference[oaicite:15]{index=15} → jerarquías todo-parte, objetos simples y compuestos tratados uniformemente.  
- **Fachada (Facade)**:contentReference[oaicite:16]{index=16} → interfaz unificada para subsistemas complejos.  

#### 🔄 De comportamiento
- **Observador (Observer)**:contentReference[oaicite:17]{index=17} → dependencia 1:n, los observadores se actualizan cuando cambia el sujeto.  
- **State**:contentReference[oaicite:18]{index=18} → encapsula estados y transiciones, comportamiento según estado.  
- **Strategy**:contentReference[oaicite:19]{index=19} → encapsula algoritmos intercambiables en tiempo de ejecución.  
- **Template Method**:contentReference[oaicite:20]{index=20} → define el esqueleto de un algoritmo delegando pasos a subclases.  
- **Comando (Command)**:contentReference[oaicite:21]{index=21} → encapsula peticiones como objetos, permite deshacer y registrar operaciones.  

---

## 🎯 Objetivos de la asignatura
- Comprender los principios de **diseño orientado a objetos**.  
- Asignar responsabilidades mediante los **patrones GRASP**.  
- Aplicar patrones de diseño (GoF) para resolver problemas recurrentes.  
- Mejorar la **cohesión**, reducir el **acoplamiento** y aumentar la **reutilización** del código.  
- Desarrollar sistemas más mantenibles, escalables y extensibles.  
