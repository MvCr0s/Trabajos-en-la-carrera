-- Reset database
--Derby does not support DROP TABLE IF EXISTS 


DROP TABLE REGISTROSDEENTREGA;

DROP TABLE LINEASDEPEDIDOS;
DROP TABLE PEDIDOS;
DROP TABLE ESTADOSDEPEDIDO;

DROP TABLE PRODUCTOS;
DROP TABLE TARJETASDEPRODUCTOS;
DROP TABLE UNIDADESDEMEDIDA;

DROP TABLE CLIENTES;

DROP TABLE DISPONIBILIDADES;
DROP TABLE VINCULACIONESCONLAEMPRESA;
DROP TABLE ROLESENEMPRESA;
DROP TABLE TIPOSDEDISPONIBILIDAD;
DROP TABLE TIPOSDEVINCULACION;
DROP TABLE TIPOSDEROL;

DROP TABLE EMPLEADOSENEGOCIOSSUSCRITOS;
DROP TABLE NEGOCIOS;
DROP TABLE DIRECCIONES;

DROP TABLE TIPOSDENEGOCIOS;

-- Enum
create table TIPOSDEROL
(
	IdTipo SMALLINT not null,
	NombreTipo VARCHAR(25) not null unique,
		PRIMARY KEY(IdTipo)
);

INSERT INTO TIPOSDEROL
VALUES  (1,'Gerente'),
        (2,'EncargadoAlmacenCocina'),
        (3,'Vendedor');

-- Enum
create table TIPOSDEVINCULACION
(
	IdTipo SMALLINT not null,
	NombreTipo VARCHAR(20) not null unique,
		PRIMARY KEY(IdTipo)

);

INSERT INTO TIPOSDEVINCULACION
VALUES  (1,'Contratado'),
        (2,'Despedido'),
        (3,'EnERTE');

-- Enum
create table TIPOSDEDISPONIBILIDAD
(
	IdTipo SMALLINT not null,
	NombreTipo VARCHAR(20) not null unique,
		PRIMARY KEY(IdTipo)
);

INSERT INTO TIPOSDEDISPONIBILIDAD
VALUES  (1,'Vacaciones'),
        (2,'BajaTemporal'),
	(3, 'Trabajando');


-- Datatype
create table DIRECCIONES
(
	Id SMALLINT not null,
	NombreDeLaVia VARCHAR(20) not null,
	Numero SMALLINT,
	Otros VARCHAR(20),
	CodigoPostal SMALLINT not null,
	Localidad VARCHAR(20) not null,
	Municipio VARCHAR(20) not null,
	Provincia VARCHAR(20) not null,
		PRIMARY KEY(Id)
);


-- Enum
create table TIPOSDENEGOCIOS
(
	IdTipo SMALLINT not null,
	NombreTipo VARCHAR(25) not null unique,
		PRIMARY KEY(IdTipo)
);

INSERT INTO TIPOSDENEGOCIOS
VALUES  (1,'PanaderiaPasteleria'),
        (2,'Carniceria'),
	(3, 'Fruteria'),
	(4,'Supermercado'),
	(5,'Cafeteria'),
	(6,'Restaurante'),
	(7,'ChocolateriaChurreria');


-- Entity
create table NEGOCIOS
(
	Cif VARCHAR(9) not null primary key,
	Nombre VARCHAR(50) not null,
	DenominacionOficial VARCHAR(50) not null,
	Direccion SMALLINT not null,
	TipoDeNegocio SMALLINT not null,
	CierreVentaAlPublico TIME not null,
	AperturaMejorNoTirarlo TIME not null,
	CierreMejorNoTirarlo TIME not null,
	FechaInscripcion DATE not null,
	Verificado BOOLEAN not null,
	Superfiable BOOLEAN not null,
		FOREIGN KEY(Direccion) REFERENCES DIRECCIONES(Id),
		FOREIGN KEY(TipoDeNegocio) REFERENCES TIPOSDENEGOCIOS(IdTipo)
);

-- Entity
create table EMPLEADOSENEGOCIOSSUSCRITOS
(
	Nif VARCHAR(9) not null primary key,
	Nombre VARCHAR(50) not null,
	Password VARCHAR(15) not null,
	Email VARCHAR(100) not null,
	Negocio VARCHAR(9) not null,
            FOREIGN KEY(Negocio) REFERENCES Negocios(Cif)
);

-- Association
create table ROLESENEMPRESA
(
	ComienzoEnRol DATE not null,
	Empleado VARCHAR(9) not null,
	Rol SMALLINT not null,
            FOREIGN KEY(Empleado) REFERENCES EMPLEADOSENEGOCIOSSUSCRITOS(Nif),
            FOREIGN KEY(Rol) REFERENCES TIPOSDEROL(IdTipo)
);

-- Association
create table VINCULACIONESCONLAEMPRESA
(
	inicio DATE not null,
	Empleado VARCHAR(9) not null,
	Vinculo SMALLINT not null,
		FOREIGN KEY(Empleado) REFERENCES EMPLEADOSENEGOCIOSSUSCRITOS(Nif),
		FOREIGN KEY(Vinculo) REFERENCES TIPOSDEVINCULACION(IdTipo) 
);

-- Association
create table DISPONIBILIDADES
(
	Comienzo DATE not null,
	FinalPrevisto DATE,
	Empleado VARCHAR(9) not null,
	Disponibilidad SMALLINT not null,
		FOREIGN KEY(Empleado) REFERENCES EMPLEADOSENEGOCIOSSUSCRITOS(Nif),
		FOREIGN KEY(Disponibilidad) REFERENCES TIPOSDEDISPONIBILIDAD(IdTipo)
);


-- Enum
create table UNIDADESDEMEDIDA
(
	IdUnidad SMALLINT not null,
	NombreUnidad VARCHAR(10) not null unique,
		PRIMARY KEY(IdUnidad)
);

INSERT INTO UNIDADESDEMEDIDA
VALUES  (1,'kilogramos'),
        (2,'litros'),
	(3, 'unidades');


-- Enum
create table ESTADOSDEPEDIDO
(
	IdEstado SMALLINT not null,
	NombreEstado VARCHAR(10) not null unique,
		PRIMARY KEY(IdEstado)
);

INSERT INTO ESTADOSDEPEDIDO
VALUES  (1,'realizado'),
        (2,'preparado'),
	(3, 'recogido'),
	(4, 'cancelado');


-- Entity
create table CLIENTES
(	Nif VARCHAR(9) not null,
		PRIMARY KEY(Nif)
);


--Entity
create table TARJETASDEPRODUCTOS
(
	Id INTEGER not null,
	Nombre VARCHAR(20) not null,
	Unidad SMALLINT not null,
	Descripcion VARCHAR(50) not null,
	Alergenos VARCHAR(100) not null,
	Ingredientes VARCHAR(100) not null,
	Negocio VARCHAR(9) not null,
                PRIMARY KEY(Id),
		FOREIGN KEY(Negocio) REFERENCES NEGOCIOS(Cif)
);

--Entity
create table PRODUCTOS
(
	Id VARCHAR(15) not null,
	Precio REAL not null,
	Fecha DATE not null,
	CantidadDisponible SMALLINT not null,
	Medida REAL not null,
	Descripcion INTEGER not null,
		PRIMARY	KEY(Id),
		FOREIGN KEY(Descripcion) REFERENCES TARJETASDEPRODUCTOS(Id)

);


--Entity
create table PEDIDOS
(
	Id VARCHAR(10) not null,
	FechaYHora TIMESTAMP not null,
	Estado SMALLINT not null,
        Cliente VARCHAR(9) not null,
        Negocio VARCHAR(9) not null,
                PRIMARY KEY(Id),
		FOREIGN KEY(Estado) REFERENCES ESTADOSDEPEDIDO(IdEstado),
                FOREIGN KEY(Cliente) REFERENCES CLIENTES(Nif),
                FOREIGN KEY(Negocio) REFERENCES NEGOCIOS(Cif)
);

--Entity
create table LINEASDEPEDIDOS
(
	Pedido VARCHAR(10) not null,
	Producto VARCHAR(15) not null,
	CantidadEnPedido SMALLINT not null,
		FOREIGN KEY(Pedido) REFERENCES PEDIDOS(Id),
		FOREIGN KEY(Producto) REFERENCES PRODUCTOS(Id)
);



--Entity, Transaction
create table REGISTROSDEENTREGA
(
	FechaYHora TIMESTAMP not null,
	Empleado VARCHAR(9) not null,
	Pedido VARCHAR(10) not null,
                FOREIGN KEY(Pedido) REFERENCES PEDIDOS(Id),
		FOREIGN KEY(Empleado) REFERENCES EMPLEADOSENEGOCIOSSUSCRITOS(Nif)
);
