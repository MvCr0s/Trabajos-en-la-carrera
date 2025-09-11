package contenedor;

import static org.junit.Assert.*;

import java.time.LocalDate;
import java.util.Set;

import org.junit.Test;
import contenedor.Contenedor.Estado;
import es.uva.inf.poo.maps.GPSCoordinate;
import muelle.Muelle;
import puerto.Puerto;
import trayecto.*;

public class TestContenedorEstandar {

    // Pruebas específicas de ContenedorEstandar
    @Test
    public void testPuedeApilarEstandar() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 800, 15, 25, true, Estado.RECOGIDA);
        assertTrue(cont.puedeApilar());
    }

    @Test
    public void testCompatibilidadInfraestructuraEstandar() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 800, 15, 25, true, Estado.RECOGIDA);
        assertTrue(cont.esCompatibleConInfraestructura("barco"));
        assertTrue(cont.esCompatibleConInfraestructura("tren"));
        assertTrue(cont.esCompatibleConInfraestructura("camion"));
        assertFalse(cont.esCompatibleConInfraestructura("avion"));
    }

    // Pruebas heredadas de Contenedor
    @Test
    public void testGetCodigo() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        assertEquals("CSQU3054383", cont.getCodigo());
    }

    @Test
    public void testSetCodigo() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        cont.setCodigo("MAEU1234567");
        assertEquals("MAEU1234567", cont.getCodigo());
    }

    @Test
    public void testGetPesoTara() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        assertEquals(1001, cont.getPesoTara(), 0);
    }

    @Test
    public void testSetPesoTara() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        cont.setPesoTara(5000);
        assertEquals(5000, cont.getPesoTara(), 0);
    }

    @Test
    public void testGetVolumenM3() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1000, 2000, 500, true, Estado.RECOGIDA);
        assertEquals(500, cont.getVolumenM3(), 0.0);
    }

    @Test
    public void testGetVolumenFt3() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1000, 2000, 500, true, Estado.RECOGIDA);
        assertEquals(500 * 35.3147, cont.getVolumenFt3(), 0.0);
    }

    @Test
    public void testSetEstadoARecogida() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1000, 2000, 500, true, Estado.TRANSITO);
        cont.setEstadoARecogida();
        assertEquals(Estado.RECOGIDA, cont.getEstado());
    }

    @Test
    public void testSetEstadoATransito() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1000, 2000, 500, true, Estado.RECOGIDA);
        cont.setEstadoATransito();
        assertEquals(Estado.TRANSITO, cont.getEstado());
    }

    @Test
    public void testConstructorConCodigoValido() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1001, 2000, 500, true, Estado.RECOGIDA);
        assertEquals("CSQU3054383", cont.getCodigo());
        assertEquals(1001, cont.getPesoTara(), 0.0);
        assertEquals(2000, cont.getcargaMaxima(), 0.0);
        assertEquals(500, cont.getVolumen(), 0.0);
        assertTrue(cont.getTecho());
        assertEquals(Estado.RECOGIDA, cont.getEstado());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConCodigoInvalido() {
        new ContenedorEstandar("INVALID123", 1001, 2000, 500, true, Estado.RECOGIDA);
    }
    
    @Test
    public void testGetPesoKg() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1000, 2000, 500, true, Contenedor.Estado.RECOGIDA);
        assertEquals(1000, cont.getPesoKg(), 0.0); // Verifica que el peso en Kg sea correcto
    }

    @Test
    public void testGetPesoLb() {
        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1000, 2000, 500, true, Contenedor.Estado.RECOGIDA);
        assertEquals(1000 * 2.20462, cont.getPesoLb(), 0.0); // Verifica que el peso en libras sea correcto
    }
    
    @Test
    public void testAddTrayecto() {
        GPSCoordinate gpsOrigen = new GPSCoordinate(50, 0);
        GPSCoordinate gpsDestino = new GPSCoordinate(30, 0);
        
        Set<String> infraestructurasValidas = Set.of("camion", "tren", "barco");

        Muelle muelleOrigen = new Muelle("01", gpsOrigen, true, 5, 10, infraestructurasValidas);
        Muelle muelleDestino = new Muelle("02", gpsDestino, true, 5, 10, infraestructurasValidas);

        Puerto puertoOrigen = new Puerto("ES-ALM");
        Puerto puertoDestino = new Puerto("ES-VLC");

        LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
        LocalDate fechaFin = LocalDate.of(2023, 10, 15);

        Trayecto trayecto = new TCamion(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin);

        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1000, 2000, 500, true, Contenedor.Estado.RECOGIDA);

        cont.addTrayecto(trayecto); // Añade el trayecto
        
        assertEquals(10193.30, cont.getPrecioTrayectos(), 0.1); // Verifica que el precio sea el esperado
    }
 
    @Test
    public void testGetPrecioTrayectosMultiples() {
        GPSCoordinate gpsOrigen = new GPSCoordinate(50, 0);
        GPSCoordinate gpsDestino = new GPSCoordinate(30, 0);
        
        Set<String> infraestructurasValidas = Set.of("camion", "tren", "barco");

        Muelle muelleOrigen = new Muelle("01", gpsOrigen, true, 5, 10, infraestructurasValidas);
        Muelle muelleDestino = new Muelle("02", gpsDestino, true, 5, 10, infraestructurasValidas);

        Puerto puertoOrigen = new Puerto("ES-ALM");
        Puerto puertoDestino = new Puerto("ES-VLC");

        LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
        LocalDate fechaFin = LocalDate.of(2023, 10, 15);

        Trayecto trayecto1 = new TCamion(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin);
        Trayecto trayecto2 = new TBarco(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin);

        ContenedorEstandar cont = new ContenedorEstandar("CSQU3054383", 1000, 2000, 500, true, Contenedor.Estado.RECOGIDA);

        cont.addTrayecto(trayecto1);
        cont.addTrayecto(trayecto2);
        

        assertEquals(30193.30, cont.getPrecioTrayectos(), 0.1); // Verifica la suma correcta de los precios de los trayectos
    }
}