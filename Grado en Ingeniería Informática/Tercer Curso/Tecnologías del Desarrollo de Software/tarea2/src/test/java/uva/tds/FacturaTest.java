package uva.tds;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.LocalDate;

import org.junit.jupiter.api.Test;

/**
 * Unit test for simple App.
 */
public class FacturaTest {

    @Test
    public void testLimiteInferior(){
        Factura f = new Factura("a", LocalDate.now(), 0);

        assertEquals("a", f.getAsunto());
        assertEquals(LocalDate.now(), f.getFecha());
        assertEquals(0, f.getImporte());
    }

    @Test
    public void testAsuntoNulo(){
        assertThrows(IllegalArgumentException.class, ()->{new Factura( null, LocalDate.now(), 0);});
    }

    @Test
    public void testAsuntoVacio(){
        assertThrows(IllegalArgumentException.class, ()->{new Factura( "    ", LocalDate.now(), 0);});
    }

    @Test
    public void testImporteMenorLimite(){
        assertThrows(IllegalArgumentException.class, ()->{new Factura( "gfg", LocalDate.now(), -1);});
    }



}
