package contenedor;

import static org.junit.Assert.*;

import org.junit.Test;
import contenedor.Contenedor.Estado;

public class TestContenedorRefrigerado {

    @Test
    public void testConstructorValidoRefrigerado() {
        ContenedorRefrigerado cont = new ContenedorRefrigerado("CSQU3054383", 1200, 25, 40, Estado.RECOGIDA);
        assertEquals("CSQU3054383", cont.getCodigo());
        assertEquals(1200, cont.getPesoTara(), 0.0);
        assertEquals(40, cont.getcargaMaxima(), 0.0);
        assertEquals(25, cont.getVolumen(), 0.0);
        assertEquals(Estado.RECOGIDA, cont.getEstado());
        assertTrue(cont.getTecho());
        assertFalse(cont.estaConectado()); // Inicialmente desconectado
    }

    @Test
    public void testConectarMotorRefrigerado() {
        ContenedorRefrigerado cont = new ContenedorRefrigerado("CSQU3054383", 1000, 20, 30, Estado.TRANSITO);
        assertFalse(cont.estaConectado()); // Inicialmente desconectado
        cont.conectar();
        assertTrue(cont.estaConectado()); // Después de conectar
    }

    @Test
    public void testDesconectarMotorRefrigerado() {
        ContenedorRefrigerado cont = new ContenedorRefrigerado("CSQU3054383", 1000, 20, 30, Estado.TRANSITO);
        cont.conectar(); // Conectar primero
        assertTrue(cont.estaConectado());
        cont.desconectar(); // Desconectar
        assertFalse(cont.estaConectado());
    }

    @Test
    public void testCompatibilidadInfraestructuraRefrigerado() {
        ContenedorRefrigerado cont = new ContenedorRefrigerado("CSQU3054383", 1000, 20, 30, Estado.TRANSITO);
        assertTrue(cont.esCompatibleConInfraestructura("barco"));
        assertTrue(cont.esCompatibleConInfraestructura("tren"));
        assertTrue(cont.esCompatibleConInfraestructura("camion"));
        assertFalse(cont.esCompatibleConInfraestructura("avion")); // Incompatible
    }

    @Test
    public void testPuedeApilarRefrigerado() {
        ContenedorRefrigerado cont = new ContenedorRefrigerado("CSQU3054383", 1000, 20, 30, Estado.TRANSITO);
        assertTrue(cont.puedeApilar()); // Un contenedor refrigerado siempre puede apilarse
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testSetTechoFalseRefrigerado() {
        ContenedorRefrigerado cont = new ContenedorRefrigerado("CSQU3054383", 1000, 20, 30, Estado.TRANSITO);
        cont.setTecho(false); // Esto debería lanzar una excepción
    }

    @Test
    public void testSetTechoTrueRefrigerado() {
        ContenedorRefrigerado cont = new ContenedorRefrigerado("CSQU3054383", 1000, 20, 30, Estado.TRANSITO);
        cont.setTecho(true); // Esto es permitido
        assertTrue(cont.getTecho());
    }
}