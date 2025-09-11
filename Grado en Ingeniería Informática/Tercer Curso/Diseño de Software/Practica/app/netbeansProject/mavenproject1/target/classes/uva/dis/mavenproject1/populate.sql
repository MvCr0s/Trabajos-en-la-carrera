
-- Direcciones (corregidos los codigos postales para SMALLINT)
INSERT INTO DIRECCIONES VALUES (1, 'Calle Mayor', 12, 'Puerta 3', 30000, 'Valladolid', 'Valladolid', 'Valladolid');
INSERT INTO DIRECCIONES VALUES (2, 'Avenida del Comercio', 45, 'Local B', 30001, 'Valladolid', 'Valladolid', 'Valladolid');

-- Negocios (sin tildes para evitar problemas de encoding)
INSERT INTO NEGOCIOS VALUES ('A12345678', 'Panaderia La Espiga', 'La Espiga S.A.', 1, 1, '20:00:00', '08:00:00', '16:00:00', '2024-01-15', true, true);
INSERT INTO NEGOCIOS VALUES ('B87654321', 'Carniceria El Chuleton', 'Carnes Selectas S.L.', 2, 2, '19:00:00', '09:00:00', '15:00:00', '2024-01-20', false, true);

-- Empleados
INSERT INTO EMPLEADOSENEGOCIOSSUSCRITOS (Nif, Nombre, Password, Email, Negocio) VALUES
('00000001A', 'Juan Perez', 'pass123', 'juan.perez@buentrigo.com', 'B87654321'),
('00000002B', 'Ana Garcia', 'anaG_24', 'ana.garcia@buentrigo.com', 'B87654321'),
('00000003C', 'Luis Rodriguez', 'Lrodriguez!', 'luis.r@carnicaslopez.es', 'B87654321'),
('00000004D', 'Maria Sanchez', 'cafeDel7', 'maria.sanchez@cafedelicias.com', 'A12345678'),
('00000005E', 'Pedro Martinez', 'superP3dr0', 'pedro.m@laesquina.es', 'A12345678'),
('00000006F', 'Clara Ruiz', 'cl4r4pass', 'clara.ruiz@panaderia.com', 'A12345678'),
('00000007G', 'Miguel Lara', 'migL@789', 'miguel.lara@carniceria.com', 'B87654321');

INSERT INTO ROLESENEMPRESA (ComienzoEnRol, Empleado, Rol) VALUES 
('2025-04-06', '00000001A', 1), 
('2025-04-06', '00000002B', 2), 
('2025-04-06', '00000003C', 3),
('2025-01-01', '00000006F', 2),
('2025-01-01', '00000007G', 1);


INSERT INTO VINCULACIONESCONLAEMPRESA (inicio, Empleado, Vinculo) VALUES
('2024-05-01', '00000001A', 1),
('2024-06-15', '00000002B', 2),
('2024-07-20', '00000003C', 3),
('2024-08-05', '00000004D', 1),
('2024-09-10', '00000005E', 1),
('2024-01-01', '00000006F', 3), 
('2024-02-01', '00000007G', 2);



INSERT INTO DISPONIBILIDADES (Comienzo, FinalPrevisto, Empleado, Disponibilidad) VALUES
('2025-04-01', '2025-04-10', '00000001A', 3),
('2025-04-03', NULL, '00000002B', 3),
('2025-04-05', NULL, '00000003C', 2),
('2025-04-07', '2025-04-20', '00000004D', 3),
('2025-04-01', NULL, '00000005E', 3),
('2025-05-01', '2025-05-10', '00000006F', 1), 
('2025-05-01', NULL, '00000007G', 2);  


INSERT INTO CLIENTES (Nif) VALUES
('11111111A'),
('22222222B'),
('33333333C'),
('44444444D'),
('55555555E'),
('66666666F');

INSERT INTO TARJETASDEPRODUCTOS (Id, Nombre, Unidad, Descripcion, Alergenos, Ingredientes, Negocio) VALUES
(1, 'Barra de pan', 3, 'Pan artesanal de trigo', 'Gluten', 'Harina, agua, levadura, sal', 'A12345678'),
(2, 'Chuleton de vaca', 1, 'Carne de vaca madurada', 'Ninguno', 'Carne de vaca', 'B87654321'),
(3, 'Bolleria surtida', 3, 'Seleccion de bolleria', 'Gluten, huevo, lactosa', 'Harina, huevo, leche, azucar', 'A12345678'),
(4, 'Tarta de queso', 3, 'Tarta casera', 'Lactosa, huevo', 'Queso, huevos, azúcar, harina', 'A12345678'),
(5, 'Morcilla', 1, 'Embutido tradicional', 'Ninguno', 'Sangre, arroz, cebolla', 'B87654321'),
(6, 'Croissant', 3, 'Croissant de mantequilla', 'Gluten, lactosa', 'Harina, mantequilla, levadura, sal', 'A12345678'),
(7, 'Pan integral', 3, 'Pan de trigo integral', 'Gluten', 'Harina integral, agua, sal, levadura', 'A12345678'),
(8, 'Donut de chocolate', 3, 'Rosquilla cubierta de chocolate', 'Gluten, huevo, lactosa', 'Harina, huevo, leche, cacao, azucar', 'A12345678'),
(9, 'Magdalena', 3, 'Magdalena esponjosa', 'Gluten, huevo, lactosa', 'Harina, huevo, leche, azucar', 'A12345678'),
(10, 'Pan sin gluten', 3, 'Pan especial para celiacos', 'Huevo', 'Harina de arroz, huevo, aceite, sal', 'A12345678'),
(11, 'Filete de ternera', 1, 'Filete fresco de ternera', 'Ninguno', 'Carne de ternera', 'B87654321'),
(12, 'Costillas de cerdo', 1, 'Costillas adobadas', 'Ninguno', 'Carne de cerdo, especias', 'B87654321'),
(13, 'Salchichas frescas', 3, 'Salchichas artesanas', 'Lactosa', 'Carne, leche en polvo, especias', 'B87654321'),
(14, 'Chorizo iberico', 2, 'Chorizo curado', 'Ninguno', 'Carne de cerdo, pimentón, ajo', 'B87654321');



INSERT INTO PRODUCTOS (Id, Precio, Fecha, CantidadDisponible, Medida, Descripcion) VALUES
('P001', 1.20, '2025-04-06', 100, 1.0, 1),
('P002', 25.50, '2025-04-06', 20, 1.5, 2),
('P003', 0.90, '2025-04-06', 50, 1.0, 3),
('P004', 2.50, '2025-05-28', 10, 1.0, 4),
('P005', 3.75, '2025-05-28', 30, 0.5, 5),
('P006', 1.10, '2025-06-05', 80, 1.0, 6),   
('P007', 1.50, '2025-06-05', 60, 1.0, 7),   
('P008', 1.75, '2025-06-05', 50, 1.0, 8),   
('P009', 1.20, '2025-06-05', 70, 1.0, 9),   
('P010', 2.00, '2025-06-05', 40, 1.0, 10),  
('P011', 12.50, '2025-06-05', 25, 0.5, 11), 
('P012', 8.90, '2025-06-05', 30, 1.0, 12),  
('P013', 6.50, '2025-06-05', 35, 0.8, 13),  
('P014', 9.80, '2025-06-05', 20, 0.7, 14);  



INSERT INTO PEDIDOS (Id, FechaYHora, Estado, Cliente, Negocio) VALUES
('PED001', '2025-05-28 09:30:00', 1, '11111111A', 'A12345678'),
('PED002', '2025-05-28 10:00:00', 2, '22222222B', 'B87654321'),
('PED003', '2025-05-28 10:30:00', 3, '33333333C', 'A12345678'),
('HOY001', '2025-06-02 09:00:00', 1, '11111111A', 'B87654321'),
('HOY002', '2025-06-02 10:30:00', 2, '22222222B', 'A12345678'),
('HOY003', '2025-06-05 11:15:00', 1, '33333333C', 'B87654321'),
('HOY004', '2025-06-02 12:45:00', 2, '33333333C', 'A12345678'),
('HOY005', '2025-06-02 13:20:00', 1, '33333333C', 'B87654321'),
('MAN001', '2025-06-02 09:00:00', 2, '22222222B', 'B87654321'),
('MAN002', '2025-06-02 10:15:00', 1, '22222222B', 'B87654321'),
('MAN003', '2025-06-02 11:45:00', 3, '11111111A', 'A12345678'),
('PED004', '2025-06-02 14:00:00', 4, '44444444D', 'B87654321'), 
('PED005', '2025-06-02 14:30:00', 3, '55555555E', 'A12345678'), 
('PED006', '2025-06-02 15:00:00', 1, '66666666F', 'A12345678');


INSERT INTO LINEASDEPEDIDOS (Pedido, Producto, CantidadEnPedido) VALUES
('MAN001', 'P002', 2),
('MAN001', 'P001', 6),
('MAN002', 'P003', 5),
('MAN003', 'P001', 4),
('HOY005', 'P002', 10),
('PED004', 'P002', 1),
('PED005', 'P004', 2),
('PED006', 'P005', 3);



INSERT INTO REGISTROSDEENTREGA (FechaYHora, Empleado, Pedido) VALUES
('2025-04-06 11:30:00', '00000001A', 'PED001'),
('2025-04-06 12:30:00', '00000003C', 'PED002'),
('2025-04-06 13:00:00', '00000004D', 'PED003'),
('2025-05-28 14:45:00', '00000001A', 'PED005');
