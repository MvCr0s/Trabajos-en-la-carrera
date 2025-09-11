# TDS 2024-2025: Entrega Final grupo 4

## Descripción
Desarollo del sistema de alquiler de bicicletas de una ciudad con varias funcionalidades, incluyendo un sistema de recompensas a los usuarios. 
Desarrollo basado en tres iteraciones que irán aumentando la funcionalidad del sistema. 

## Roadmap
1. Primera iteración: desarrollará las funcionalidades F1-F7 del documento .
2. Segunda iteración: desarrollará Ruta y GestorRutas así como persistencia al añadir gestores en aislamiento.
3. Tercera iteración: implementará las interfaces descritas en la anterior iteración con sus correspondientes clases de test. Además se incluirá la integración continua y la relación con una base de datos mediante Hibernate. 
Finalmente se ha llevado a cabo la refactorización del código utilizando sonarqube para detectar Code Smells y el catálogo de Fowler para mejorar el mismo. 

## Tiempo total en horas-persona
1. Primera iteración: 320h 12m = 13d 8h 12m
2. Segunda iteración: 120h 51m = 5d 51m
3. Tercera iteración: 

- **Total: 441h 3m = 18d 9h 3m**

## Autores
- Ainhoa Carbajo
- Emily Rodrigues
- Marcos De Diego

## Clases que forman parte de la solución
1. Creadas durante la primera iteración:
- Alquiler
- Bicicleta
- Bloqueo
- GestorParadas
- GestorRecompensas
- GestorUsuario
- Parada
- Recompensa
- Reserva
- Usuario

2. Creadas durante la segunda iteración:
- GestorParadasEnAislamiento
- IParadaRepositorio
- GestorRecompensasEnAislamiento
- IRecompensaRepositorio
- GestorUsuarioEnAislamiento
- IUsuarioRepositorio
- Ruta
- GestorRutas
- ICalculoRuta (autor: marcorr)
- GestorRutaEnAislamiento
- IRutaRepositorio

3. Creadas durante la tercera iteración:
- ParadaRepositorio
- UsuarioRepositorio
- RecompensaRepositorio
- RutaRepositorio

## Refactorizaciones del catálogo de Fowler
- Dead Code / Duplicated Code: ambas refactorizaciones se han aplicado en clases como Usuario, Recompensa, GestorParadas ... con la intención de limpiar el código de líneas repetidas o inaccesibles.
- Extract Variable: en todas las clases de test y en las constantes en las clases base se han empleado variables para simplificar el código. 
- Change Unidirectional Association to Bidirectional: aunque no explícitamente en un solo commit, esta refactorización se ha llevado a cabo para gestionar los pares de entidades Usuario-recompensa y Parada-Bicicletas. Así se han añadido atributos (parada en bicicleta) y métodos (setUsuario en Recompensa) para adaptarse a los cambios
- Consolidate Conditional Expression: refactorización muy empleada en la gestión de Usuario-Recompensa para mejorar la sintaxis de comprobación de los métodos. 
- Extract Method: a lo largo de todo el proyecto, se ha ido disminuyendo el tamaño de los métodos de cada clase de cada paquete, para ayudar en la comprensión y depuración del código.
- (NO) Replace Method with Method Object: si bien en las clases que implementan las interfaces de los repositorios hay métodos comunes (como clearDatabase), no se ha creído conveniente realizar dicha refactorización. En concreto, clearDatabase debería limpiar todas las tablas para que funcione bien con cada implementación, lo cual afectaría negativamente la eficiencia de las implementaciones que no usan todas las tablas, generando operaciones redundantes e innecesarias.