package uva.dis.tests;

public class LoginTest {
/**
    private static DBConnection db;

    @BeforeAll
    public static void setUpClass() throws Exception {
        // Inicializar la base de datos y ejecuta los scripts de creación y población.
        db = DBConnection.getInstance();
        db.openConnection();
    }

    @AfterAll
    public static void tearDownClass() throws Exception {
        if (db != null) {
            db.closeConnection();
        }
    }

    @Test
    public void testIdentificacionExitosa() {
        // Supongamos que populate.sql inserta un empleado activo con:
        // NIF "00000001A", password "pass123" y nombre "Juan Pérez"
        EmpleadoDTO nombre = ControladorCUIdentificarse.comprobarIdentificacion("00000001A", "pass123");
        assertNotNull(nombre, "El empleado debería identificarse correctamente.");
        assertEquals("Juan Perez", nombre, "El nombre del empleado debería ser 'Juan Perez'.");
    }

    @Test
    public void testIdentificacionFallida() {
        // Si se introducen credenciales incorrectas, se espera que se retorne null.
        EmpleadoDTO resultado = ControladorCUIdentificarse.comprobarIdentificacion("12345678A", "wrongPassword");
        assertNull(resultado, "La identificación debe fallar con credenciales incorrectas.");
    }
     
    * */
}

