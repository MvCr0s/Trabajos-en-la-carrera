-- Autores:  ALMARAZ ARRANZ, MARCOS / CARBAJO ORGAZ, AINHOA / DE DIEGO MARTIN, MARCOS / PENA MARQUES, RODRIGO 
-- Fecha Creacion: Noviembre 2024
-- Descripcion: Base de datos creada para cumplir los requisitos de la "ORDENANZA REGULADORA DE LA OCUPACIÓN DE LA VÍA PÚBLICA DE MALAGA"


-- Drop tables 
DROP TABLE Peticion;
DROP TABLE Toldo;
DROP TABLE Infraccion;
DROP TABLE Autorizacion1y2;
DROP TABLE Titularidad;
DROP TABLE Autorizacion;
DROP TABLE Elemento;
DROP TABLE Establecimiento;
DROP TABLE Titular;

-- Drop types
DROP TYPE tipoInfraccion;
DROP TYPE motivo;
DROP TYPE modalidadOcupacion ;
DROP TYPE tipoElemento;
DROP TYPE tipoAnclaje;
DROP TYPE tipoActividad;
DROP TYPE tipoEvento;
DROP TYPE tipoAutorizacion;
DROP TYPE tipoTitular;


CREATE TYPE tipoInfraccion AS ENUM ('leve', 'grave', 'muyGrave'); 

CREATE TYPE motivo AS ENUM ( 
    'nuevaLicencia', 
    'regularizacion', 
    'inicioExplotacion', 
    'cambioTitularidad' 
); 

CREATE TYPE modalidadOcupacion AS ENUM ( 
    'anual', 
    'temporal', 
    'ocasionalSemanaSanta', 
    'ocasionalFeria' 
); 


CREATE TYPE tipoElemento AS ENUM( 
'silla', 
'mesa', 
'sombrilla', 
'cortavientos', 
'jardinera', 
'celosia', 
'toldo', 
'otro', 
'voladizo', 
'calefactor' 
); 
 
CREATE TYPE tipoAnclaje AS ENUM ('abatible', 'rigido'); 

CREATE TYPE tipoActividad AS ENUM ( 
    'expositor', 
    'cartel', 
    'reclamoPublicitario', 
    'rampaAcceso', 
    'actividadDivulgativa', 
    'vehiculoPromocional', 
    'otro', 
    'rodajeCinematografico' 
); 

CREATE TYPE tipoEvento AS ENUM ( 
    'feriaAgosto', 
    'feriaCentro', 
    'fiestasTradicionales', 
    'otros' 
); 

CREATE TYPE tipoAutorizacion AS ENUM ( 
    'tipo1',
    'tipo2'
); 


CREATE TYPE tipoTitular AS ENUM ('particular', 'empresa'); 

CREATE TABLE Titular( 
    idf VARCHAR(50), 
    nombre VARCHAR(255) NOT NULL, 
    direccion VARCHAR(255) NOT NULL, 
    telefono VARCHAR(9) NOT NULL, 
    correo VARCHAR(255) NOT NULL, 
    tipo tipoTitular NOT NULL,  
    fechaRegistro DATE NOT NULL, 
    poliza BOOLEAN NOT NULL, 
    PRIMARY KEY (idf), 

CONSTRAINT check_telefono 
CHECK( telefono ~ '^[0-9]+$') 
); 

CREATE TABLE Establecimiento ( 
    isd VARCHAR(50) , 
    nombre VARCHAR(255) NOT NULL, 
    direccion VARCHAR(255) NOT NULL, 
    superficie DOUBLE PRECISION NOT NULL, 
    descripcion VARCHAR(255) NOT NULL, 
    ubicacion VARCHAR(255) NOT NULL, 
    alumbradoElectrico BOOLEAN NOT NULL, 
    PRIMARY KEY (isd) 

); 


CREATE TABLE Autorizacion ( 
    idAutorizacion VARCHAR(50),
    idf VARCHAR(50), 
    isd VARCHAR(50),
    descripcion VARCHAR(255) NOT NULL, 
    motivoFueraPlazo motivo, 
    fechaEmision DATE NOT NULL, 
    fechaAdmision DATE NOT NULL, 
    ubicacion VARCHAR(255) NOT NULL, 
    fechaInicioVigencia DATE NOT NULL, 
    fechaFinVigencia DATE NOT NULL, 
    inicioHorarioAutorizado TIME NOT NULL, 
    finHorarioAutorizado TIME NOT NULL, 
    superficieAutorizada DOUBLE PRECISION NOT NULL, 
    superficieSolicitada DOUBLE PRECISION NOT NULL, 
    planAprovechamiento BOOLEAN NOT NULL, 
    publicidad BOOLEAN NOT NULL, 
    tipoActividad tipoActividad, 
    tipoEvento tipoEvento,  
    fechaInicioMontaje DATE, 
    fechaFinMontaje DATE, 

    PRIMARY KEY (idAutorizacion),  
    FOREIGN KEY (idf) REFERENCES Titular(idf) ON DELETE CASCADE ON UPDATE CASCADE, 
    FOREIGN KEY (isd) REFERENCES Establecimiento(isd) ON DELETE CASCADE ON UPDATE CASCADE, 
 
-- Restricción: 
    CONSTRAINT check_tipo
    CHECK ((tipoActividad IS NULL AND isd IS NULL AND tipoEvento is NOT NULL AND fechaFinMontaje is NOT NULL AND fechaInicioMontaje is NOT NULL AND idf is NOT NULL) 
    OR (tipoEvento IS NULL AND fechaInicioMontaje IS NULL AND fechaFinMontaje IS NULL AND tipoActividad is NOT NULL AND isd IS NOT NULL AND idf is NULL) 
    OR (tipoActividad IS NULL AND tipoEvento IS NULL AND fechaInicioMontaje IS NULL AND fechaFinMontaje IS NULL AND isd is NULL and idf is NULL)),

-- Restricción la fechaAdmision no se ha producido más de 3 meses después de la fechaEmision 
    CONSTRAINT check_fechaAdmision 
        CHECK (fechaAdmision <= fechaEmision + INTERVAL '3 months'), 

    CONSTRAINT check_fechas 
    CHECK (fechaAdmision > fechaEmision), 


    CONSTRAINT check_vigencia 
    CHECK (fechaFinVigencia > fechaInicioVigencia), 

    CONSTRAINT check_montaje 
    CHECK (  fechaFinMontaje >    fechaInicioMontaje), 

-- Restricción: las autorizaciones  de tipo 1 y 2 solo se pueden formular en septiembre, octubre y noviembre
    CONSTRAINT check_fechas_autorizacion
    CHECK ((motivoFueraPlazo IS NULL OR
            (EXTRACT(MONTH FROM fechaEmision) NOT IN (9, 10, 11) OR
            EXTRACT(YEAR FROM fechaEmision) <> EXTRACT(YEAR FROM fechaInicioVigencia) - 1))
    )

); 


CREATE TABLE Infraccion ( 
        idInfraccion VARCHAR(50), 
        tipo tipoInfraccion NOT NULL, 
        fechaEmision DATE NOT NULL, 
        sancion DOUBLE PRECISION NOT NULL, 
        estadoPago BOOLEAN NOT NULL, 
        motivo VARCHAR(250) NOT NULL, 
        idAutorizacion VARCHAR(50), 
        PRIMARY KEY (idInfraccion), 
        FOREIGN KEY (idAutorizacion) REFERENCES Autorizacion(idAutorizacion) ON DELETE CASCADE ON UPDATE CASCADE, 

 
-- Restricción: los valores de sanción dependen del tipo de infracción 
CONSTRAINT check_tipoCantidadSancion 
    CHECK ( 
        (tipo = 'leve' AND sancion <= 750) OR 
        (tipo = 'grave' AND sancion BETWEEN 751 AND 1500) OR 
        (tipo = 'muyGrave' AND sancion BETWEEN 1501 AND 3000) 
    ) 
); 

CREATE TABLE Autorizacion1y2 ( 
    idAutorizacion VARCHAR(50) , 
    renovacion BOOLEAN NOT NULL, 
    tarifa DOUBLE PRECISION NOT NULL, 
    fianza DOUBLE PRECISION NOT NULL, 
    modalidadOcupacion modalidadOcupacion NOT NULL , 
    isd VARCHAR(20), 
    tipo tipoAutorizacion NOT NULL,
    PRIMARY KEY (idAutorizacion), 
    FOREIGN KEY (idAutorizacion) REFERENCES autorizacion(idAutorizacion) ON DELETE CASCADE ON UPDATE CASCADE, 
    FOREIGN KEY (isd) REFERENCES Establecimiento(isd) ON DELETE CASCADE ON UPDATE CASCADE,

    CONSTRAINT check_tipo
    check (tipo='tipo1' OR(tipo='tipo2' AND modalidadOcupacion='anual'))
);

 

 CREATE TABLE Titularidad ( 
      fechaInicio DATE NOT NULL, 
      isd VARCHAR(50), 
      idf VARCHAR(50), 
      FOREIGN KEY (isd) REFERENCES Establecimiento(isd) ON DELETE CASCADE ON UPDATE CASCADE, 
      FOREIGN KEY (idf) REFERENCES Titular(idf) ON DELETE CASCADE ON UPDATE CASCADE 

); 


 

CREATE TABLE Elemento ( 
    codigoReferencia VARCHAR(50), 
    tipoElemento tipoElemento NOT NULL,  
    ancho DOUBLE PRECISION NOT NULL, 
    alto DOUBLE PRECISION NOT NULL, 
    material VARCHAR(100) NOT NULL, 
    tarifa DOUBLE PRECISION NOT NULL, 
    descripcion VARCHAR(255) NOT NULL, 
    color VARCHAR(50) NOT NULL, 
    PRIMARY KEY (codigoReferencia), 


CONSTRAINT check_altura_elemento 
    CHECK ( 
        (tipoElemento NOT IN ('cortavientos', 'celosia'))  
        OR (alto <= 1.5) 
    ) 

); 

 

 

CREATE TABLE Toldo ( 
    codigoReferencia VARCHAR(50) , 
    homologado BOOLEAN NOT NULL, 
    anclaje tipoAnclaje NOT NULL,  
    cerramiento BOOLEAN NOT NULL, 
    publicidad BOOLEAN NOT NULL, 
    descripcionAnclaje VARCHAR(255), 
    tratamientoIgnifugo BOOLEAN NOT NULL, 
    PRIMARY KEY (codigoReferencia), 
    FOREIGN KEY (codigoReferencia) REFERENCES Elemento(codigoReferencia) ON DELETE CASCADE ON UPDATE CASCADE, 


-- Restricción para color y tratamiento ignífugo 
    CONSTRAINT check_toldo_color_tratamiento 
    CHECK (tratamientoIgnifugo = TRUE  ) 
); 


 CREATE TABLE Peticion ( 
   cantidad INTEGER NOT NULL, 
   codigoReferencia VARCHAR(50), 
   idAutorizacion VARCHAR(50), 
   FOREIGN KEY (codigoReferencia) REFERENCES Elemento(codigoReferencia) ON DELETE CASCADE ON UPDATE CASCADE, 
   FOREIGN KEY (idAutorizacion) REFERENCES Autorizacion(idAutorizacion) ON DELETE CASCADE ON UPDATE CASCADE

); 

-- Restricción: si modalidadOcupacion es 'anual' o 'temporal', renovacion debe ser TRUE 
ALTER TABLE Autorizacion1y2 
ADD CONSTRAINT check_ModalidadOcupacion 
    CHECK ( 
        modalidadOcupacion NOT IN ('anual', 'temporal') OR renovacion = TRUE 

    ); 
    

-- Inserts para la tabla Titular
INSERT INTO Titular (idf, nombre, direccion, telefono, correo, tipo, fechaRegistro, poliza)
VALUES 
('12345678A', 'Juan Pérez', 'Calle Falsa 123, Málaga', '912345678', 'juan.perez@example.com', 'particular', '2023-01-10', TRUE),
('23456789B', 'Comercial Gómez S.L.', 'Avenida del Comercio 45, Málaga', '954123456', 'contacto@gomezsl.com', 'empresa', '2022-11-20', TRUE),
('34567890C', 'Ana Sánchez', 'Calle Real 56, Málaga', '961234567', 'ana.sanchez@example.com', 'particular', '2023-03-15', FALSE),
('45678901D', 'Tech Solutions S.A.', 'Parque Empresarial Norte, Málaga', '958765432', 'info@techsolutions.com', 'empresa', '2023-04-01', TRUE),
('56789012E', 'Fernando López', 'Calle Alameda Principal 1, Málaga', '952123456', 'fernando.lopez@example.com', 'particular', '2023-05-20', TRUE),
('67890123F', 'Restaurantes del Sur S.L.', 'Avenida de Andalucía 22, Málaga', '952987654', 'contacto@restsur.com', 'empresa', '2023-06-15', TRUE),
('78901234G', 'Beatriz Gutiérrez', 'Calle Larios 10, Málaga', '951234567', 'beatriz.gutierrez@example.com', 'particular', '2023-07-01', FALSE),
('89012345H', 'Ingeniería del Sol S.L.', 'Calle Martínez 14, Málaga', '951765432', 'info@ingenieriasol.com', 'empresa', '2023-08-01', TRUE);

-- Inserts para la tabla Establecimiento
INSERT INTO Establecimiento (isd, nombre, direccion, superficie, descripcion, ubicacion, alumbradoElectrico)
VALUES 
('E001', 'Bar La Esquina', 'Plaza Mayor 10, Málaga', 120.5, 'Bar de tapas con terraza', 'Plaza Mayor', TRUE),
('E002', 'Restaurante El Puerto', 'Calle del Mar 8, Málaga', 250.0, 'Restaurante especializado en mariscos', 'Zona Puerto', FALSE),
('E003', 'Chiringuito El Sol', 'Paseo Marítimo 20, Málaga', 180.0, 'Chiringuito frente a la playa', 'Playa', TRUE),
('E004', 'Café Central', 'Calle Larios 5, Málaga', 100.0, 'Cafetería céntrica', 'Centro Ciudad', TRUE),
('E005', 'Restaurante del Sur', 'Avenida de Andalucía 22, Málaga', 300.0, 'Restaurante de cocina mediterránea', 'Avenida Andalucía', FALSE),
('E006', 'Bar El Pescador', 'Calle del Río 15, Málaga', 90.0, 'Bar de pescados y mariscos', 'Zona Pesquera', TRUE);

-- Inserts para la tabla Autorizacion
INSERT INTO Autorizacion (idAutorizacion, idf, isd, descripcion, motivoFueraPlazo, fechaEmision, fechaAdmision, ubicacion, fechaInicioVigencia, fechaFinVigencia, inicioHorarioAutorizado, finHorarioAutorizado, superficieAutorizada, superficieSolicitada, planAprovechamiento, publicidad, tipoActividad, tipoEvento, fechaInicioMontaje, fechaFinMontaje)
VALUES
('A001', NULL, NULL, 'Autorización de terraza', NULL, '2022-09-01', '2022-09-15', 'Plaza Mayor', '2023-06-01', '2023-09-01', '10:00:00', '23:00:00', 100.0, 120.0, TRUE, FALSE, NULL, NULL, NULL, NULL),
('A002', NULL, NULL, 'Autorización de terraza', NULL, '2022-10-01', '2022-11-10', 'Zona Puerto', '2023-04-01', '2023-04-30', '09:00:00', '21:00:00', 200.0, 220.0, FALSE, TRUE, NULL, NULL, NULL, NULL),
('A003', NULL, 'E002', 'Autorización para actividad de expositor', NULL, '2023-08-01', '2023-08-05', 'Zona Puerto', '2023-09-01', '2023-12-01', '11:00:00', '22:00:00', 150.0, 160.0, TRUE, FALSE, 'expositor', NULL, NULL, NULL),
('A004', '34567890C', NULL, 'Autorización para feria tradicional (tipo4)', NULL, '2023-10-01', '2023-10-10', 'Plaza Mayor', '2023-11-01', '2023-11-05', '10:00:00', '18:00:00', 100.0, 110.0, FALSE, TRUE, NULL, 'fiestasTradicionales', '2023-11-01', '2023-11-05'),
('A005', '67890123F', NULL, 'Autorización para feria Agosto (tipo4)', NULL, '2023-06-01', '2023-06-10', 'Avenida Andalucía', '2023-04-01', '2023-12-31', '09:00:00', '23:00:00', 200.0, 250.0, TRUE, FALSE, NULL,'feriaAgosto', '2023-8-01', '2023-8-30'),
('A006', NULL, 'E003', 'Autorización para montaje temporal de eventos actualizada', NULL, '2023-09-01', '2023-09-05', 'Playa', '2023-10-01', '2023-10-15', '10:00:00', '20:00:00', 180.0, 200.0, FALSE, TRUE, 'actividadDivulgativa', NULL, NULL, NULL),
('A007', NULL, NULL, 'Autorización de ampliación de terraza actualizada', NULL, '2023-08-15', '2023-08-20', 'Zona Pesquera', '2023-09-01', '2023-12-31', '10:00:00', '23:00:00', 90.0, 100.0, TRUE, TRUE, NULL, NULL, NULL, NULL),
('A008', NULL, NULL, 'Autorización fuera de plazo', 'cambioTitularidad', '2020-12-01', '2021-02-01', 'Plaza Mayor', '2023-06-01', '2023-09-01', '10:00:00', '23:00:00', 100.0, 120.0, TRUE, FALSE, NULL, NULL, NULL, NULL);

-- Inserts para la tabla Infraccion
INSERT INTO Infraccion (idInfraccion, tipo, fechaEmision, sancion, estadoPago, motivo, idAutorizacion)
VALUES
('I001', 'leve', '2023-06-15', 500, FALSE, 'No respetar los horarios autorizados', 'A001'),
('I002', 'grave', '2023-07-01', 1000, TRUE, 'Instalación de elementos no autorizados', 'A002');

-- Inserts para la tabla Autorizacion1y2
INSERT INTO Autorizacion1y2 (idAutorizacion, renovacion, tarifa, fianza, modalidadOcupacion, isd, tipo)
VALUES
('A001', TRUE, 1500.0, 500.0, 'anual', 'E001', 'tipo2'),
('A002', TRUE, 1000.0, 300.0, 'temporal', 'E002', 'tipo1'),
('A007', TRUE, 1000.0, 300.0, 'temporal', 'E005', 'tipo1'),
('A008', TRUE, 1000.0, 300.0, 'temporal', 'E001', 'tipo1');

-- Inserts para la tabla Titularidad
INSERT INTO Titularidad (fechaInicio, isd, idf)
VALUES
('2023-01-01', 'E001', '12345678A'),
('2022-01-01', 'E001', '34567890C'),
('2021-01-01', 'E001', '56789012E'),
('2023-02-01', 'E002', '23456789B'),
('2023-05-15', 'E005', '67890123F'),
('2023-09-01', 'E003', '56789012E'),
('2023-10-01', 'E006', '56789012E');

-- Inserts para la tabla Elemento
INSERT INTO Elemento (codigoReferencia, tipoElemento, ancho, alto, material, tarifa, descripcion, color)
VALUES
('EL001', 'silla', 0.5, 1.0, 'plástico', 10.0, 'Silla para terraza', 'rojo'),
('EL002', 'mesa', 1.0, 0.8, 'madera', 20.0, 'Mesa para terraza', 'marrón'),
('EL003', 'toldo', 2.0, 2.5, 'lona', 50.0, 'Toldo abatible para terraza', 'azul'),
('EL004', 'cortavientos', 1.5, 1.5, 'vidrio', 30.0, 'Cortavientos para terraza', 'transparente'),
('EL005', 'toldo', 3.0, 2.5, 'lona', 70.0, 'Toldo para terraza', 'transparente');

-- Inserts para la tabla Toldo
INSERT INTO Toldo (codigoReferencia, homologado, anclaje, cerramiento, publicidad, descripcionAnclaje, tratamientoIgnifugo)
VALUES
('EL003', TRUE, 'abatible', TRUE, FALSE, 'Anclaje a pared', TRUE),
('EL005', TRUE, 'rigido', FALSE, TRUE, 'Anclaje al suelo', TRUE);

-- Inserts para la tabla Peticion
INSERT INTO Peticion (cantidad, codigoReferencia, idAutorizacion)
VALUES
(10, 'EL001', 'A002'),
(5, 'EL002', 'A002'),
(1, 'EL003', 'A001'),
(1, 'EL005', 'A001'),
(6, 'EL002', 'A007');


-- Consulta que muestra la autorizacion con su titular y establecimiento (si lo tiene)
SELECT 
    t.idf AS idTitular, 
    t.nombre AS nombreTitular, 
    a.idAutorizacion AS idAutorizacion,
    NULL AS nombreEstablecimiento
FROM 
    Autorizacion a
INNER JOIN 
    Titular t 
ON 
    a.idf = t.idf

UNION

SELECT
    t.idf AS idTitular, 
    t.nombre AS nombreTitular, 
    a.idAutorizacion AS idAutorizacion,
    e.nombre AS nombreEstablecimiento
FROM 
    Autorizacion a
INNER JOIN 
    Establecimiento e 
ON 
    a.isd = e.isd
INNER JOIN 
    Titularidad ti 
ON 
    e.isd = ti.isd
AND 
    ti.fechaInicio = (
        SELECT MAX(ti2.fechaInicio)
        FROM Titularidad ti2
        WHERE ti2.isd = ti.isd
    )
INNER JOIN 
    Titular t 
ON 
    ti.idf = t.idf

UNION

SELECT
    t.idf AS idTitular, 
    t.nombre AS nombreTitular, 
    a1y2.idAutorizacion AS idAutorizacion,
    e.nombre AS nombreEstablecimiento
FROM 
    Autorizacion1y2 a1y2
INNER JOIN 
    Establecimiento e 
ON 
    a1y2.isd = e.isd
INNER JOIN 
    Titularidad ti 
ON 
    e.isd = ti.isd
AND 
    ti.fechaInicio = (
        SELECT MAX(ti2.fechaInicio)
        FROM Titularidad ti2
        WHERE ti2.isd = ti.isd
    )
INNER JOIN 
    Titular t 
ON 
    ti.idf = t.idf;

-- Autorizaciones Títulos I  con titular y establecimiento
SELECT
    t.idf AS idTitular, 
    t.nombre AS nombreTitular, 
    a1y2.idAutorizacion AS idAutorizacion,
    e.nombre AS nombreEstablecimiento
FROM 
    Autorizacion1y2 a1y2
INNER JOIN 
    Establecimiento e 
ON 
    a1y2.isd = e.isd
INNER JOIN 
    Titularidad ti 
ON 
    e.isd = ti.isd
AND 
    ti.fechaInicio = (
        SELECT MAX(ti2.fechaInicio)
        FROM Titularidad ti2
        WHERE ti2.isd = ti.isd
    )
INNER JOIN 
    Titular t 
ON 
    ti.idf = t.idf
WHERE
    a1y2.tipo = 'tipo1';



-- Autorizaciones Títulos II  con titular y establecimiento
SELECT 
    t.idf AS idTitular, 
    t.nombre AS nombreTitular, 
    a1y2.idAutorizacion AS idAutorizacion,
    e.nombre AS nombreEstablecimiento
FROM 
    Autorizacion1y2 a1y2
INNER JOIN 
    Establecimiento e 
ON 
    a1y2.isd = e.isd
INNER JOIN 
    Titularidad ti 
ON 
    e.isd = ti.isd
AND 
    ti.fechaInicio = (
        SELECT MAX(ti2.fechaInicio)
        FROM Titularidad ti2
        WHERE ti2.isd = ti.isd
    )
INNER JOIN 
    Titular t 
ON 
    ti.idf = t.idf
WHERE
    a1y2.tipo = 'tipo2';



-- Autorizaciones Títulos III  con titular y establecimiento

SELECT 
    t.idf AS idTitular, 
    t.nombre AS nombreTitular, 
    a.idAutorizacion AS idAutorizacion,
    e.nombre AS nombreEstablecimiento
FROM 
    Autorizacion a
INNER JOIN 
    Establecimiento e 
ON 
    a.isd = e.isd
INNER JOIN 
    Titularidad ti 
ON 
    e.isd = ti.isd
AND 
    ti.fechaInicio = (
        SELECT MAX(ti2.fechaInicio)
        FROM Titularidad ti2
        WHERE ti2.isd = ti.isd
    )
INNER JOIN 
    Titular t 
ON 
    ti.idf = t.idf;


-- Autorizaciones Títulos IV  con titular y establecimiento

SELECT 
    t.idf AS idTitular, 
    t.nombre AS nombreTitular, 
    a.idAutorizacion AS idAutorizacion
FROM 
    Autorizacion a
INNER JOIN 
    Titular t 
ON 
    a.idf = t.idf;



--Lista de titulares y sus titularidades 
SELECT 
    t.idf AS idTitular,
    t.nombre AS nombreTitular,
    tit.fechaInicio AS fechainicio,
    e.nombre AS nombreEstablecimiento
FROM
    Titular t
INNER JOIN
    Titularidad tit
ON
    t.idf=tit.idf
INNER JOIN
    Establecimiento e
ON
    e.isd=tit.isd;

--Historial de titulares del bar La Esquina
SELECT 
    t.idf AS idTitular,
    t.nombre AS nombreTitular,
    tit.fechaInicio AS fechainicio
FROM
    Titular t
INNER JOIN
    Titularidad tit
ON
    t.idf=tit.idf
INNER JOIN
    Establecimiento e
ON
    e.isd=tit.isd
WHERE e.isd = 'E001';

--Historial de autorizaciones del bar El Puerto
 SELECT
    a.descripcion AS nombreAutorizacion,
    CASE
        WHEN a.tipoEVEnto IS NOT NULL THEN 'tipo3' ELSE 'tipo4'
    END AS tipoAutorizacion,
    a.fechaAdmision AS fechaAdmision
FROM
    Autorizacion a
INNER JOIN
    Establecimiento e
ON
    a.isd = e.isd
WHERE
    e.isd = 'E002'

UNION ALL

SELECT
    a.descripcion AS nombreAutorizacion,
    a1y2.tipo::TEXT AS tipoAutorizacion,
    a.fechaAdmision AS fechaAdmision
FROM
    Autorizacion a
INNER JOIN
    Autorizacion1y2 a1y2
ON
    a.idAutorizacion=a1y2.idAutorizacion
INNER JOIN
    Establecimiento e
ON
    a1y2.isd = e.isd
WHERE
    e.isd = 'E002';

--Autorizaciones fuera de plazo
 SELECT
    a.descripcion AS nombreAutorizacion,
    CASE
        WHEN a.tipoEVEnto IS NOT NULL THEN 'tipo3' ELSE 'tipo4'
    END AS tipoAutorizacion,
    a.motivoFueraPlazo:: TEXT as motivo,
    a.fechaAdmision AS fechaAdmision
FROM
    Autorizacion a
INNER JOIN
    Establecimiento e
ON
    a.isd = e.isd
WHERE 
    a.motivoFueraPlazo is not null

UNION ALL

SELECT
    a.descripcion AS nombreAutorizacion,
    a1y2.tipo::TEXT AS tipoAutorizacion,
     a.motivoFueraPlazo:: TEXT as motivo,
    a.fechaAdmision AS fechaAdmision
FROM
    Autorizacion a
INNER JOIN
    Autorizacion1y2 a1y2
ON
    a.idAutorizacion=a1y2.idAutorizacion
INNER JOIN
    Establecimiento e
ON
    a1y2.isd = e.isd
WHERE 
    a.motivoFueraPlazo is not null;

--Historial de infracciones
 SELECT
    a.descripcion AS nombreAutorizacion,
    i.tipo AS tipoInfraccion,
    i.fechaEmision as fecha,
    i.motivo as motivo
FROM
   Infraccion i
INNER JOIN
    Autorizacion a
ON
    i.idAutorizacion = a.idAutorizacion;
