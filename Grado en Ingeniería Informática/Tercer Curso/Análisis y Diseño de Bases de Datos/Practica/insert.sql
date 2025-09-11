1º Titular Movil Incorrecto
INSERT INTO Titular VALUES ('18090996r','Rodrigo','CalleHuertas Nº 30','61678440r','rodrigo.pena.marques@gmail.com','particular','2024-10-19',TRUE);
2º Titular Movil Correcto
INSERT INTO Titular VALUES ('18090996r','Rodrigo','CalleHuertas Nº 30','61678440r','rodrigo.pena.marques@gmail.com','particular','2024-10-19',TRUE);

3ºEstablecimientoCorrecto
INSERT into Establecimiento VALUES ('1234','ElRodris','CalleHuertas Nº 30', 100,'Bar de ambiente','Valladolid',FALSE);

4ºAutorizacion Tipo Incorrecta
INSERT into autorizacion VALUES ('123456789',NULL,'1234', 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,'expositor',NULL,NULL,'2024-12-31');

INSERT into autorizacion VALUES ('123456789',NULL,'1234', 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,'expositor',NULL,'2024-12-01','2024-12-31');

INSERT into autorizacion VALUES ('123456789',NULL,'1234', 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,'expositor','feriaAgosto','2024-12-01','2024-12-31');

INSERT into autorizacion VALUES ('123456789',NULL,'1234', 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,'expositor','feriaAgosto',NULL,'2024-12-31');

INSERT into autorizacion VALUES ('123456789','18090996r','1234', 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,'expositor','feriaAgosto',NULL,'2024-12-31');

INSERT into autorizacion VALUES ('123456789','18090996r','1234', 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,'expositor','feriaAgosto',NULL,NULL);

INSERT into autorizacion VALUES ('123456789','18090996r',NULL, 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,'expositor','feriaAgosto',NULL,NULL);


INSERT into autorizacion VALUES ('123456789','18090996r',NULL, 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,NULL,'feriaAgosto',NULL,NULL);

INSERT into autorizacion VALUES ('123456789','18090996r',NULL, 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,NULL,NULL,'2024-12-31',NULL);



5ºAutorizacion Tipo Correcta
INSERT into autorizacion VALUES ('123456789',NULL,'1234', 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,'expositor',NULL,NULL,NULL);

INSERT into autorizacion VALUES ('123456789','18090996r',NULL, 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,NULL,'feriaAgosto','2024-12-01','2024-12-31');

INSERT into autorizacion VALUES ('123456789',NULL,'1234', 'AutorizacionPrueba1','1234','2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,NULL,NULL,NULL,NULL);





7º fechaMontaje Incorrecta
INSERT into autorizacion VALUES ('123456789','18090996r',NULL, 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,NULL,'feriaAgosto','2024-12-30','2024-11-30');

8º fechas Incorrecta
INSERT into autorizacion VALUES ('123456789','18090996r',NULL, 'AutorizacionPrueba1',NULL,'2024-12-31','2024-12-01','Valladolid','2025-12-01','2025-12-31','08:00:00','18:00:00',100,120, TRUE,FALSE,NULL,'feriaAgosto','2024-11-30','2024-12-30');

9ºfechas montaje Incorrecta
INSERT into autorizacion VALUES ('123456789','18090996r',NULL, 'AutorizacionPrueba1',NULL,'2024-12-01','2024-12-31','Valladolid','2025-12-31','2025-12-01','08:00:00','18:00:00',100,120, TRUE,FALSE,NULL,'feriaAgosto','2024-11-30','2024-12-31');

10ºfechasAdmisionIncorrectas
INSERT into autorizacion VALUES ('123456789','18090996r',NULL, 'AutorizacionPrueba1',NULL,'2024-6-01','2024-12-31','Valladolid','2025-12-31','2025-12-01','08:00:00','18:00:00',100,120, TRUE,FALSE,NULL,'feriaAgosto','2024-11-30','2024-12-31');

11ºInfraccionIncorrecta
INSERT into infraccion Values
('1234','leve','2024-11-30',751,FALSE,'ocupacion Ilegal','123456789');


INSERT into infraccion Values
('123456789','grave','2024-11-30',750,FALSE,'ocupacion Ilegal','123456789');

INSERT into infraccion Values
('1234','grave','2024-11-30',1501,FALSE,'ocupacion Ilegal','123456789');

INSERT into infraccion Values
('1234','muyGrave','2024-11-30',1500,FALSE,'ocupacion Ilegal','123456789');

INSERT into infraccion Values
('1234','muyGrave','2024-11-30',3001,FALSE,'ocupacion Ilegal','123456789');

10ºInfraccionCorrecta
INSERT into infraccion Values
('1234','leve','2024-11-30',750,FALSE,'ocupacion Ilegal','123456789');

INSERT into infraccion Values
('12345','grave','2024-11-30',1500,FALSE,'ocupacion Ilegal','123456789');

INSERT into infraccion Values
('12346','muyGrave','2024-11-30',3000,FALSE,'ocupacion Ilegal','123456789');

12ºAutorizacion1y2Correcta
INSERT into Autorizacion1y2 Values
('12346787',TRUE,3000,3000,'anual','1234');


13ºElementoIncorrecto
INSERT into Elemento Values
('2','cortavientos',0.5,1.6,'madera',50,'sila normal','rojo');
INSERT into Elemento Values
('2','celosia',0.5,1.6,'madera',50,'sila normal','rojo');

13ºElementoCorrecto
INSERT into Elemento Values
('1','silla',0.5,1.5,'madera',50,'sila normal','rojo');

-- Restricción: si modalidadOcupacion es 'anual' o 'temporal', renovacion debe ser TRUE Comprobada

-- Crear función para validar la fechaSolicitud según modalidadOcupacion La he quitado del todo por que daba errores graves

-- Si hay plan de aprovechamiento, reducir la tarifa a la mitad NO tengo claro que haya que hacer esto

  -- Validar finHorarioAutorizado No funciona pero tampoco tengo claro que haya que hacerlo

-- Validar si existe un toldo con color 'transparente' Comprobada

 -- Verificar si el tipo de elemento es 'toldo' y la modalidad de ocupación no es 'anual' No funciona
