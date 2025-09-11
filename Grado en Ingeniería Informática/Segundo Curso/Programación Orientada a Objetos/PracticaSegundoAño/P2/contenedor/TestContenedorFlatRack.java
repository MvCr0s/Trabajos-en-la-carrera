package contenedor;

import static org.junit.Assert.*;

import org.junit.Test;
import contenedor.Contenedor.Estado;

public class TestContenedorFlatRack {

    @Test
    public void testConstructorValidoFlatRack() {
        ContenedorFlatRack cont = new ContenedorFlatRack("CSQU3054383", 1500, 50, 30, Estado.TRANSITO);
        assertEquals("CSQU3054383", cont.getCodigo());
        assertEquals(1500, cont.getPesoTara(), 0.0);
        assertEquals(50, cont.getcargaMaxima(), 0.0);
        assertEquals(30, cont.getVolumen(), 0.0);
        assertEquals(Estado.TRANSITO, cont.getEstado());
        assertFalse(cont.getTecho()); // FlatRack no tiene techo
    }

    @Test
    public void testNoApilableFlatRack() {
        ContenedorFlatRack cont = new ContenedorFlatRack("CSQU3054383", 1500, 50, 30, Estado.TRANSITO);
        assertFalse(cont.puedeApilar());
    }

    @Test
    public void testCompatibilidadInfraestructuraFlatRack() {
        ContenedorFlatRack cont = new ContenedorFlatRack("CSQU3054383", 1500, 50, 30, Estado.TRANSITO);
        assertTrue(cont.esCompatibleConInfraestructura("barco"));
        assertTrue(cont.esCompatibleConInfraestructura("tren"));
        assertFalse(cont.esCompatibleConInfraestructura("camion")); // Incompatible
        assertFalse(cont.esCompatibleConInfraestructura("avion")); // Incompatible
    }

    @Test
    public void testPlazasRequeridasFlatRack() {
        ContenedorFlatRack cont = new ContenedorFlatRack("CSQU3054383", 1500, 50, 30, Estado.TRANSITO);
        assertEquals(2, cont.plazasRequeridas);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testSetTechoTrueFlatRack() {
        ContenedorFlatRack cont = new ContenedorFlatRack("CSQU3054383", 1500, 50, 30, Estado.TRANSITO);
        cont.setTecho(true); // Esto debería lanzar una excepción
    }

    @Test
    public void testSetTechoFalseFlatRack() {
        ContenedorFlatRack cont = new ContenedorFlatRack("CSQU3054383", 1500, 50, 30, Estado.TRANSITO);
        cont.setTecho(false); // Esto es permitido
        assertFalse(cont.getTecho());
    }
}