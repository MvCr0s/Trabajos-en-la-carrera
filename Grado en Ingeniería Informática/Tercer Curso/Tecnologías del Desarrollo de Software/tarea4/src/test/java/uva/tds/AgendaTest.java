package uva.tds;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AgendaTest {

    private Agenda agenda;

    @BeforeEach
    void setUp() {
        agenda = new Agenda();
    }

    @Test
    void testAgregarContactoValido() {
        Contacto contacto = new Contacto("Juan", "Pérez");
        agenda.addContacto(contacto);
        Contacto resultado = agenda.getContacto("Juan");
        assertNotNull(resultado);
        assertEquals("Pérez", resultado.getApellido());
    }

    @Test
    void testAgregarContactoNulo() {
        assertThrows(IllegalArgumentException.class, () -> {
            agenda.addContacto(null);
        });
    }


    @Test
    void testGetContactoExistente() {
        Contacto contacto = new Contacto("Ana", "Gómez");
        agenda.addContacto(contacto);
        Contacto resultado = agenda.getContacto("Ana");
        assertNotNull(resultado);
        assertEquals("Gómez", resultado.getApellido());
    }

	@Test
	void testGetContactoNoExistente() {
		Contacto resultado = agenda.getContacto("Desconocido");
		assertNotNull(resultado); 
		assertEquals("Desconocido", resultado.getNombre());
		assertEquals("Desconocido", resultado.getApellido());
		assertTrue(resultado instanceof ContactoNulo); 
	}
	

    @Test
    void testModificarApellidoExistente() {
        Contacto contacto = new Contacto("Luis", "Fernández");
        agenda.addContacto(contacto);
        agenda.modificarApellido("Luis", "García");
        Contacto resultado = agenda.getContacto("Luis");
        assertNotNull(resultado);
        assertEquals("García", resultado.getApellido());
    }

    @Test
    void testModificarApellidoContactoNoExistente() {
        assertThrows(IllegalArgumentException.class, () -> {
            agenda.modificarApellido("NoExiste", "NuevoApellido");
        });
    }

    @Test
    void testModificarApellidoInvalido() {
        Contacto contacto = new Contacto("Carlos", "Martínez");
        agenda.addContacto(contacto);
        assertThrows(IllegalArgumentException.class, () -> {
            agenda.modificarApellido("Carlos", null);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            agenda.modificarApellido("Carlos", "");
        });
    }
}
