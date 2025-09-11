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

    superficie FLOAT NOT NULL, 

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

    motivoFueraPlazo VARCHAR(255), 

    fechaEmision DATE NOT NULL, 

    fechaAdmision DATE NOT NULL, 

    ubicacion VARCHAR(255) NOT NULL, 

    fechaInicioVigencia DATE NOT NULL, 

    fechaFinVigencia DATE NOT NULL, 

    inicioHorarioAutorizado TIME NOT NULL, 

    finHorarioAutorizado TIME NOT NULL, 

    superficieAutorizada FLOAT  NOT NULL, 

    superficieSolicitada FLOAT NOT NULL, 

    planAprovechamiento BOOLEAN NOT NULL, 

    publicidad BOOLEAN NOT NULL, 

    tipoActividad tipoActividad, 

    tipoEvento tipoEvento,  

    fechaInicioMontaje DATE, 

    fechaFinMontaje DATE, 

    PRIMARY KEY (idAutorizacion),  
    FOREIGN KEY (idf) REFERENCES Titular(idf), 
    FOREIGN KEY (isd) REFERENCES Establecimiento(isd), 


 

 
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

CHECK (  fechaFinMontaje >    fechaInicioMontaje) 

); 

 

 

 

 

CREATE TABLE Infraccion ( 

        idInfraccion VARCHAR(50), 

        tipo tipoInfraccion NOT NULL, 

        fechaEmision DATE NOT NULL, 

        sancion FLOAT NOT NULL, 

        estadoPago BOOLEAN NOT NULL, 

        motivo VARCHAR(250) NOT NULL, 

        idAutorizacion VARCHAR(50), 

        PRIMARY KEY (idInfraccion), 

        FOREIGN KEY (idAutorizacion) REFERENCES Autorizacion(idAutorizacion), 

 

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
    
    tipo tipoAutorizacion NOT NULL;

    PRIMARY KEY (idAutorizacion), 

    FOREIGN KEY (idAutorizacion) REFERENCES autorizacion(idAutorizacion), 

    FOREIGN KEY (isd) REFERENCES Establecimiento(isd) 
	
    CONSTRAINT check_tipo
    check (tipo='tipo2' AND modalidadOcupacion='anual')
);

 

 CREATE TABLE Titularidad ( 

      fechaInicio DATE NOT NULL, 

      isd VARCHAR(50), 

      idf VARCHAR(50), 

      FOREIGN KEY (isd) REFERENCES Establecimiento(isd), 

      FOREIGN KEY (idf) REFERENCES Titular(idf) 

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

    FOREIGN KEY (codigoReferencia) REFERENCES Elemento(codigoReferencia), 

 

-- Restricción para color y tratamiento ignífugo 

    CONSTRAINT check_toldo_color_tratamiento 

    CHECK (tratamientoIgnifugo = TRUE  ) 
); 


 CREATE TABLE Peticion ( 

   Numero INTEGER NOT NULL, 

   codigoReferencia VARCHAR(50), 

   idAutorizacion VARCHAR(50), 

   FOREIGN KEY (codigoReferencia) REFERENCES Elemento(codigoReferencia), 

   FOREIGN KEY (idAutorizacion) REFERENCES Autorizacion(idAutorizacion) 

); 

-- Restricción: si modalidadOcupacion es 'anual' o 'temporal', renovacion debe ser TRUE 

ALTER TABLE Autorizacion1y2 

ADD CONSTRAINT check_ModalidadOcupacion 

    CHECK ( 

        modalidadOcupacion NOT IN ('anual', 'temporal') OR renovacion = TRUE 

    ); 


-- Asignar el trigger a la tabla 

CREATE TRIGGER validar_tarifa 

BEFORE INSERT OR UPDATE ON  

Autorizacion1y2 

FOR EACH ROW 

EXECUTE FUNCTION validar_tarifa(); 

 

-- Función para validar la restricción del toldo 

CREATE OR REPLACE FUNCTION validar_toldo() 

RETURNS TRIGGER AS $$ 

BEGIN 

    -- Validar si existe un toldo con color 'transparente' 

    IF EXISTS ( 

        SELECT 1 

        FROM Elemento E 

        JOIN Toldo T ON E.codigoReferencia = T.codigoReferencia 

        WHERE E.codigoReferencia = NEW.codigoReferencia 

          AND E.color = 'transparente' 

          AND T.tratamientoIgnifugo = FALSE -- Agregar condiciones adicionales si es necesario 

    ) THEN 

        RAISE EXCEPTION 'El toldo de color transparente debe tener tratamiento ignífugo.'; 

    END IF; 

 

    -- Si pasa la validación, devolver la fila 

    RETURN NEW; 

END; 

$$ LANGUAGE plpgsql; 
