package contenedor;

import static org.junit.Assert.*;

import java.time.LocalDate;

import org.junit.Test;

import contenedor.Contenedor.Estado;
import es.uva.inf.poo.maps.GPSCoordinate;
import muelle.Muelle;
import puerto.Puerto;
import trayecto.Trayecto;

public class TestContenedor {

	@Test
	public void testGetCodigo() {
		Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
		assertEquals("CSQU3054383",cont.getCodigo());
	}
	
	@Test
	public void testSetCodigo() {
		Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
		cont.setCodigo("MSCU1234567");
		assertEquals("MSCU1234567",cont.getCodigo());
	}
	
	@Test
	public void testGetPeso_Tara() {
		Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
		assertEquals(1001,cont.getPeso_tara(),0);
	}
	
	@Test
	public void testSetPeso_Tara() {
		Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
		cont.setPeso_tara(5000);
		assertEquals(5000,cont.getPeso_tara(),0);
	}
	
	@Test
    public void testGetCargaMaxima() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        cont.setCarga_maxima(5000.0);
        assertEquals(5000.0, cont.getCarga_maxima(), 0.0);
    }

    @Test
    public void testSetCargaMaxima() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        cont.setCarga_maxima(2500.0);
        assertEquals(2500.0, cont.getCarga_maxima(), 0.0);
    }

    @Test
    public void testGetVolumen() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        cont.setVolumen(200.0);
        assertEquals(200.0, cont.getVolumen(), 0.0);
    }

    @Test
    public void testSetVolumen() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        cont.setVolumen(300.0);
        assertEquals(300.0, cont.getVolumen(), 0.0);
    }

    @Test
    public void testGetTecho() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        assertTrue(cont.getTecho());
    }

    @Test
    public void testSetTecho() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        cont.setTecho(false);
        assertFalse(cont.getTecho());
    }

    @Test
    public void testGetEstado() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        assertEquals(Estado.RECOGIDA, cont.getEstado());
    }

    @Test
    public void testSetEstado() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Estado.RECOGIDA);
        cont.setEstado(Estado.TRANSITO);
        assertEquals(Estado.TRANSITO, cont.getEstado());
    }
    
    @Test
    public void testConstructorConCodigoValido() {
            Contenedor cont = new Contenedor("CSQU3054383", 1001, 2000, 500, true, Estado.RECOGIDA);
            assertEquals("CSQU3054383", cont.getCodigo());
            assertEquals(1001, cont.getPeso_tara(), 0.0);
            assertEquals(2000, cont.getCarga_maxima(), 0.0);
            assertEquals(500, cont.getVolumen(), 0.0);
            assertTrue(cont.getTecho());
            assertEquals(Estado.RECOGIDA, cont.getEstado());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConCodigoInvalido1() {
        Contenedor cont = new Contenedor("INVALID123", 1001, 2000, 500, true, Estado.RECOGIDA);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConCodigoInvalido2() {
        Contenedor cont = new Contenedor("CSQU3054384", 1001, 2000, 500, true, Estado.RECOGIDA);
    }
    
    @Test
    public void testSetEstadoARecogida() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 2000, 500, true, Estado.TRANSITO);
        cont.setEstadoARecogida();
        assertEquals(Estado.RECOGIDA, cont.getEstado());
    }

    @Test
    public void testSetEstadoATransito() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 2000, 500, true, Estado.RECOGIDA);
        cont.setEstadoATransito();
        assertEquals(Estado.TRANSITO, cont.getEstado());
    }

    @Test
    public void testGetVolumenM3() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 2000, 500, true, Estado.RECOGIDA);
        assertEquals(500, cont.getVolumenM3(), 0.0);
    }

    @Test
    public void testGetVolumenFt3() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 2000, 500, true, Estado.RECOGIDA);
        assertEquals(500 * 35.3147, cont.getVolumenFt3(), 0.0);
    }

    @Test
    public void testGetPesoKg() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 2000, 500, true, Estado.RECOGIDA);
        assertEquals(1001, cont.getPesoKg(), 0.0);
    }

    @Test
    public void testGetPesoLb() {
        Contenedor cont = new Contenedor("CSQU3054383", 1001, 2000, 500, true, Estado.RECOGIDA);
        assertEquals(1001 * 2.20462, cont.getPesoLb(), 0.0);
    }
    
    @Test
    public void testGetPrecioTrayectos() {
        GPSCoordinate gpsOrigen = new GPSCoordinate(50, 0);
        GPSCoordinate gpsDestino = new GPSCoordinate(30, 0);

        Muelle muelleOrigen = new Muelle("01", gpsOrigen, true, 5, 10);
        Muelle muelleDestino = new Muelle("02", gpsDestino, true, 5, 10);

        Puerto puertoOrigen = new Puerto("ES-ALM");
        Puerto puertoDestino = new Puerto("ES-VLC");

        LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
        LocalDate fechaFin = LocalDate.of(2023, 10, 15);

        Trayecto trayecto1 = new Trayecto(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin, 50.0, 10.0);
        Trayecto trayecto2 = new Trayecto(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin, 70.0, 15.0);

        Contenedor cont = new Contenedor("CSQU3054383", 1001, 1002, 1003, true, Contenedor.Estado.RECOGIDA);

        cont.addTrayecto(trayecto1);
        cont.addTrayecto(trayecto2);

        double precioTotal = cont.getPrecioTrayectos();

        double precioEsperado = 0;
		try {
			precioEsperado = trayecto1.getPrecio() + trayecto2.getPrecio();
		} catch (NoSuchFieldException e) {
			e.printStackTrace();
		}
        assertEquals(precioEsperado, precioTotal, 0.01);
    }
	

}
