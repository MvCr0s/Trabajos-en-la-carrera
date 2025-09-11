package trayecto;

import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.Test;

import java.time.LocalDate;
import es.uva.inf.poo.maps.GPSCoordinate;
import puerto.Puerto;
import muelle.Muelle;

public class TrayectoTest {
    
    private Muelle muelleOrigen;
    private Muelle muelleDestino;
    private Puerto puertoOrigen;
    private Puerto puertoDestino;
    private Trayecto trayecto;
    
    @Before
    public void setUp() {
        GPSCoordinate gpsOrigen = new GPSCoordinate(50, 0);
        GPSCoordinate gpsDestino = new GPSCoordinate(30, 0); 
        
        muelleOrigen = new Muelle("01", gpsOrigen, true, 5, 10);
        muelleDestino = new Muelle("02", gpsDestino, true, 5, 10);
        
        puertoOrigen = new Puerto("ES-ALM");
        puertoDestino = new Puerto("ES-VLC");

        LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
        LocalDate fechaFin = LocalDate.of(2023, 10, 15);

        trayecto = new Trayecto(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin);
    }
    

    @Test
    public void testTrayectoMuelleOrigen() {
        assertEquals(muelleOrigen, trayecto.getMuelleOrigen());
    }

    @Test
    public void testTrayectoPuertoOrigen() {
        assertEquals(puertoOrigen, trayecto.getPuertoOrigen());
    }

    @Test
    public void testTrayectoFechaInicio() {
        assertEquals(LocalDate.of(2023, 10, 10), trayecto.getFechaInicio());
    }

    @Test
    public void testTrayectoMuelleDestino() {
        assertEquals(muelleDestino, trayecto.getMuelleDestino());
    }

    @Test
    public void testTrayectoPuertoDestino() {
        assertEquals(puertoDestino, trayecto.getPuertoDestino());
    }

    @Test
    public void testTrayectoFechaFin() {
        assertEquals(LocalDate.of(2023, 10, 15), trayecto.getFechaFin());
    }
    
    @Test
    public void testSetMuelleOrigen() {
        Muelle nuevoMuelleOrigen = new Muelle("03", new GPSCoordinate(40, 0), true, 6, 10);
        trayecto.setMuelleOrigen(nuevoMuelleOrigen);
        assertEquals(nuevoMuelleOrigen, trayecto.getMuelleOrigen());
    }

    @Test
    public void testSetPuertoOrigen() {
        Puerto nuevoPuertoOrigen = new Puerto("ES-BCN");
        trayecto.setPuertoOrigen(nuevoPuertoOrigen);
        assertEquals(nuevoPuertoOrigen, trayecto.getPuertoOrigen());
    }

    @Test
    public void testSetFechaInicio() {
        LocalDate nuevaFechaInicio = LocalDate.of(2023, 11, 1);
        trayecto.setFechaInicio(nuevaFechaInicio);
        assertEquals(nuevaFechaInicio, trayecto.getFechaInicio());
    }

    @Test
    public void testSetMuelleDestino() {
        Muelle nuevoMuelleDestino = new Muelle("04", new GPSCoordinate(45, 0), true, 7, 12);
        trayecto.setMuelleDestino(nuevoMuelleDestino);
        assertEquals(nuevoMuelleDestino, trayecto.getMuelleDestino());
    }

    @Test
    public void testSetPuertoDestino() {
        Puerto nuevoPuertoDestino = new Puerto("FR-MRS");
        trayecto.setPuertoDestino(nuevoPuertoDestino);
        assertEquals(nuevoPuertoDestino, trayecto.getPuertoDestino());
    }

    @Test
    public void testSetFechaFin() {
        LocalDate nuevaFechaFin = LocalDate.of(2023, 12, 1);
        trayecto.setFechaFin(nuevaFechaFin);
        assertEquals(nuevaFechaFin, trayecto.getFechaFin());
    }
    
    
    
    
    
    @Test
    public void constructor2() throws NoSuchFieldException {

        LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
        LocalDate fechaFin = LocalDate.of(2023, 10, 15);

        trayecto = new Trayecto(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin, 100, 100);
        assertEquals(trayecto.getPrecio(),trayecto.calcularPrecioTrayecto(100, 100),0);
    }
    
    @Test(expected = NoSuchFieldException.class)
    public void constructor1SetPrecio() throws NoSuchFieldException {

        LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
        LocalDate fechaFin = LocalDate.of(2023, 10, 15);

        trayecto = new Trayecto(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin);
		assertEquals(trayecto.getPrecio(),trayecto.calcularPrecioTrayecto(100, 100),0);
    }
    
    
    
    
    @Test
    public void testEsFechaFinSuperior() {
        LocalDate fecha = LocalDate.of(2023, 10, 12);
        assertTrue(trayecto.esFechaFinSuperior(fecha));
    }

    @Test
    public void testEsFechaFinSuperiorMismaFecha() {
        LocalDate fecha = LocalDate.of(2023, 10, 15);
        assertFalse(trayecto.esFechaFinSuperior(fecha));
    }
    
    @Test
    public void testEsFechaFinSuperiorAnterior() {
        LocalDate fecha = LocalDate.of(2023, 10, 16);
        assertFalse(trayecto.esFechaFinSuperior(fecha));
    }


    @Test
    public void testCalcularPrecioTrayecto() {
        double costePorDia = 50.0; 
        double costePorMillaMarina = 10.0; 
        double precioEsperado = (5 * 50.0) + (trayecto.obtenerDistanciaMillasMarinas() * 10.0);
        assertEquals(precioEsperado, trayecto.calcularPrecioTrayecto(costePorDia, costePorMillaMarina), 0.01);
    }

    @Test
    public void testObtenerDistanciaMillasMarinas() {
        double distancia = trayecto.obtenerDistanciaMillasMarinas();
        assertTrue(distancia > 0);
    }
    
    @Test
    public void testObtenerDistanciaMillasMarinasExacta() {
        double distanciaEsperada =  1378.71;
        assertEquals(distanciaEsperada, trayecto.obtenerDistanciaMillasMarinas(),2); 
    }

    @Test
    public void testObtenerInformacionCompleta() {
        String infoEsperada = "Trayecto desde " + puertoOrigen.getLocalidad() + " (" + puertoOrigen.getPais() + ")" + " en el muelle " + muelleOrigen + " el " + LocalDate.of(2023, 10, 10) + " hasta " + puertoDestino.getLocalidad() + " (" + puertoDestino.getPais() + ")" + " en el muelle " + muelleDestino + " el " + LocalDate.of(2023, 10, 15);
        assertEquals(infoEsperada, trayecto.obtenerInformacionCompleta());
    }
    

    @Test(expected = IllegalArgumentException.class)
    public void testCalcularPrecioTrayectoCostoDiaNegativo() {
        double costePorDia = -10.0; 
        double costePorMillaMarina = 10.0; 
        trayecto.calcularPrecioTrayecto(costePorDia, costePorMillaMarina);
    }


    @Test(expected = IllegalArgumentException.class)
    public void testCalcularPrecioTrayectoCostoMillaMarinaNegativo() {
        double costePorDia = 50.0; 
        double costePorMillaMarina = -5.0;
        trayecto.calcularPrecioTrayecto(costePorDia, costePorMillaMarina);
    }


    @Test(expected = IllegalArgumentException.class)
    public void testFechaFinAnteriorFechaInicio() {
        LocalDate fechaInicio2 = LocalDate.of(2023, 10, 15);
        LocalDate fechaFin2 = LocalDate.of(2023, 10, 10); 
        new Trayecto(muelleOrigen, puertoOrigen, fechaInicio2, muelleDestino, puertoDestino, fechaFin2);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testPuertoOrigenIgualDestino() {
	   	 LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
	   	 LocalDate fechaFin = LocalDate.of(2023, 10, 15);
	   	 
    	Puerto puertoOrigen2 = new Puerto("ES-ALM");
        Puerto puertoDestino2 = new Puerto("ES-ALM");
    	
        new Trayecto(muelleOrigen, puertoOrigen2, fechaInicio, muelleDestino, puertoDestino2, fechaFin);
    }
    
    
    @Test(expected = IllegalArgumentException.class)
    public void testMuelleOrigenIgualDestino() {
    	 LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
    	 LocalDate fechaFin = LocalDate.of(2023, 10, 15);
    	
    	Muelle muelleOrigen2 = new Muelle("01", new GPSCoordinate(50, 0), true, 5, 10);
        Muelle muelleDestino2 = new Muelle("01", new GPSCoordinate(50, 0), true, 5, 10);
    	
        new Trayecto(muelleOrigen2, puertoOrigen, fechaInicio, muelleDestino2, puertoDestino, fechaFin);
    }


    
}
