package uva.tds;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;


import java.time.LocalDate;
import java.util.ArrayList;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class GestorFacturasTest {

    private GestorFacturas g;
    private Factura f;
    private Factura f2;
    private Factura esperadaf;
    private Factura esperadaf2;
    private ArrayList<Factura> esperado;
    private ArrayList<Factura> d;

    @BeforeEach
    public void startUp(){
        g = new GestorFacturas(LocalDate.now(), LocalDate.now().plusDays(0), "a");
        f = new Factura("a", LocalDate.now().plusDays(5), 0);
        f2 = new Factura("atgr", LocalDate.now().plusDays(10), 0);

        esperadaf = new Factura("a", LocalDate.now().plusDays(5), 0);
        esperadaf2 =  new Factura("atgr", LocalDate.now().plusDays(10), 0);
        esperado = new ArrayList<>();
        d = new ArrayList<>();

    }

    @Test
    public void testConstructorLimiteInferior(){
        GestorFacturas g = new GestorFacturas(LocalDate.now(), LocalDate.now().plusDays(0), "a");

        assertEquals(LocalDate.now(), g.getFechaInicio());
        assertEquals(LocalDate.now().plusDays(0), g.getFechaFin());
        assertEquals("a", g.getNombre());
        assertTrue( g.isEstado());
    }

    @Test
    public void testConstructorLimiteSuperior(){
        GestorFacturas g = new GestorFacturas(LocalDate.now(), LocalDate.now().plusDays(0), "0123456789");

        assertEquals(LocalDate.now(), g.getFechaInicio());
        assertEquals(LocalDate.now().plusDays(0), g.getFechaFin());
        assertEquals("0123456789", g.getNombre());
        assertTrue( g.isEstado());
    }


    @Test
    public void testConstructorNombreMenorLimiteInferior(){
       assertThrows(IllegalArgumentException.class,()->{ new GestorFacturas(LocalDate.now(), LocalDate.now().plusDays(0), " ");});
    }

    @Test
    public void testConstructorNombreNull(){
       assertThrows(IllegalArgumentException.class,()->{ new GestorFacturas(LocalDate.now(), LocalDate.now().plusDays(0), null);});
    }

    @Test
    public void testConstructorNombreMayorLimiteSuperior(){
       assertThrows(IllegalArgumentException.class,()->{ new GestorFacturas(LocalDate.now(), LocalDate.now().plusDays(0), "12345678910");});
    }

    @Test
    public void testConstructorFechaInicioMayorFechaFin(){
       assertThrows(IllegalArgumentException.class,()->{ new GestorFacturas(LocalDate.of(2025, 4, 19), LocalDate.of(2024, 4, 19), "12345678910");});
    }

    @Test
    public void testAñadirFactura(){
        g.añadir(f);
        esperado.add(esperadaf);

        assertIterableEquals(g.getFacturas(), esperado);
        assertIterableEquals(g.listarFacturasPorFecha(), esperado);
    }

    @Test
    public void testAñadirFacturaNull(){
       assertThrows(IllegalArgumentException.class,()-> {g.añadir(null);} );
    }


    @Test
    public void testAñadirFacturaRepetida(){
        g.añadir(f);
       assertThrows(IllegalStateException.class,()-> {g.añadir(esperadaf);} );
    }

    @Test
    public void testAñadirFacturaFechasMal(){
        Factura f2 = new Factura("asd", LocalDate.now().plusDays(-2), 5);
       assertThrows(IllegalStateException.class,()-> {g.añadir(f2);} );
    }
    

    @Test 
    public void testañadirFacturaCerrada(){
        g.setEstado(false);
        assertThrows(IllegalStateException.class,()-> {g.añadir(f);} );
    }

    @Test
    public void testAñadirListaFactura(){
        d.add(f);
        d.add(f2);
        g.añadirFacturas(d);

        esperado.add(esperadaf);
        esperado.add(esperadaf2);

        assertIterableEquals(g.getFacturas(), esperado);
        assertIterableEquals(g.listarFacturasPorFecha(), esperado);
        assertIterableEquals(g.listarFacturasPorImporte(), esperado);
    }

    @Test
    public void testAñadirListaFacturaRepe(){
        d.add(f);
        d.add(f);
        assertThrows(IllegalStateException.class, ()->{g.añadirFacturas(d);});
    }

    @Test
    public void testAñadirListaFacturaNull(){
        assertThrows(IllegalArgumentException.class, ()->{g.añadirFacturas(null);});
    }

    @Test
    public void testAñadirListaFacturaUnaNull(){
        d.add(f);
        d.add(null);
        assertThrows(IllegalArgumentException.class, ()->{g.añadirFacturas(d);});
    }
    


}
