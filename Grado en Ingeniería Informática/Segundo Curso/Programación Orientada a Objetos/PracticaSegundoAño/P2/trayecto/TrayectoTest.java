package trayecto;

import static org.junit.Assert.*;

import java.time.LocalDate;
import java.util.HashSet;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import contenedor.ContenedorRefrigerado;
import contenedor.Contenedor.Estado;
import es.uva.inf.poo.maps.GPSCoordinate;
import muelle.Muelle;
import puerto.Puerto;

public class TrayectoTest {
	private Muelle muelleOrigen;
    private Muelle muelleDestino;
    private Puerto puertoOrigen;
    private Puerto puertoDestino;
    private TCamion trayectoCamion;
    private ContenedorRefrigerado contenedorRefrigerado;
    
    Set<String> infraestructurasAdmitidas = new HashSet<>();

    @Before
    public void setUp() {
        GPSCoordinate gpsOrigen = new GPSCoordinate(50, 0);
        GPSCoordinate gpsDestino = new GPSCoordinate(30, 0); 
        
        infraestructurasAdmitidas.add("barco");
        infraestructurasAdmitidas.add("tren");
        infraestructurasAdmitidas.add("camion");
        

        muelleOrigen = new Muelle("01", gpsOrigen, true, 5, 10, infraestructurasAdmitidas);
        muelleDestino = new Muelle("02", gpsDestino, true, 5, 10, infraestructurasAdmitidas);
        
        puertoOrigen = new Puerto("ES-ALM");
        puertoDestino = new Puerto("ES-VLC");

        LocalDate fechaInicio = LocalDate.of(2023, 10, 10);
        LocalDate fechaFin = LocalDate.of(2023, 10, 15);
        

        contenedorRefrigerado = new ContenedorRefrigerado("CSQU3054383", 1200, 25, 40, Estado.RECOGIDA);
        
        muelleOrigen.asignarContenedorAPlaza(contenedorRefrigerado, 1);

        trayectoCamion = new TCamion(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin);
    }
    
    @Test
    public void testTrayectoMuelleOrigen() {
        assertEquals(muelleOrigen, trayectoCamion.getMuelleOrigen());
    }

    @Test
    public void testTrayectoPuertoOrigen() {
        assertEquals(puertoOrigen, trayectoCamion.getPuertoOrigen());
    }

    @Test
    public void testTrayectoFechaInicio() {
        assertEquals(LocalDate.of(2023, 10, 10), trayectoCamion.getFechaInicio());
    }

    @Test
    public void testTrayectoMuelleDestino() {
        assertEquals(muelleDestino, trayectoCamion.getMuelleDestino());
    }

    @Test
    public void testTrayectoPuertoDestino() {
        assertEquals(puertoDestino, trayectoCamion.getPuertoDestino());
    }

    @Test
    public void testTrayectoFechaFin() {
        assertEquals(LocalDate.of(2023, 10, 15), trayectoCamion.getFechaFin());
    }

    @Test
    public void testSetMuelleOrigen() {
        Muelle nuevoMuelleOrigen = new Muelle("03", new GPSCoordinate(40, 0), true, 6, 10,  infraestructurasAdmitidas);
        trayectoCamion.setMuelleOrigen(nuevoMuelleOrigen);
        assertEquals(nuevoMuelleOrigen, trayectoCamion.getMuelleOrigen());
    }

    @Test
    public void testSetPuertoOrigen() {
        Puerto nuevoPuertoOrigen = new Puerto("ES-BCN");
        trayectoCamion.setPuertoOrigen(nuevoPuertoOrigen);
        assertEquals(nuevoPuertoOrigen, trayectoCamion.getPuertoOrigen());
    }

    @Test
    public void testSetFechaInicio() {
        LocalDate nuevaFechaInicio = LocalDate.of(2023, 11, 1);
        trayectoCamion.setFechaInicio(nuevaFechaInicio);
        assertEquals(nuevaFechaInicio, trayectoCamion.getFechaInicio());
    }

    @Test
    public void testSetMuelleDestino() {
        Muelle nuevoMuelleDestino = new Muelle("04", new GPSCoordinate(45, 0), true, 7, 12, infraestructurasAdmitidas);
        trayectoCamion.setMuelleDestino(nuevoMuelleDestino);
        assertEquals(nuevoMuelleDestino, trayectoCamion.getMuelleDestino());
    }

    @Test
    public void testSetPuertoDestino() {
        Puerto nuevoPuertoDestino = new Puerto("FR-MRS");
        trayectoCamion.setPuertoDestino(nuevoPuertoDestino);
        assertEquals(nuevoPuertoDestino, trayectoCamion.getPuertoDestino());
    }

    @Test
    public void testSetFechaFin() {
        LocalDate nuevaFechaFin = LocalDate.of(2023, 12, 1);
        trayectoCamion.setFechaFin(nuevaFechaFin);
        assertEquals(nuevaFechaFin, trayectoCamion.getFechaFin());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetFechaFinAnteriorFechaInicio() {
        LocalDate fechaFin2 = LocalDate.of(2023, 10, 5);
        trayectoCamion.setFechaFin(fechaFin2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetMuelleDestinoIgualOrigen() {
        trayectoCamion.setMuelleDestino(muelleOrigen);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetPuertoDestinoIgualOrigen() {
        trayectoCamion.setPuertoDestino(puertoOrigen);
    }

    @Test
    public void testCalcularPrecioTrayecto() {
      
        double precioEsperado = trayectoCamion.calcularPrecioTrayecto();
        assertTrue(precioEsperado > 0);
    }

    @Test
    public void testObtenerDistanciaMillasMarinas() {
        double distancia = trayectoCamion.obtenerDistanciaMillasMarinas();
        assertTrue(distancia > 0);
    }
    

    @Test
    public void testObtenerInformacionCompleta() {
        String infoEsperada = "Trayecto simple desde " + trayectoCamion.getPuertoOrigen().getLocalidad() + " hasta " + trayectoCamion.getPuertoDestino().getLocalidad()  + " (TCamion)";
        assertEquals(infoEsperada, trayectoCamion.obtenerInformacionCompleta());
    }
    
    @Test
    public void testGetTipoTransporte() {
        assertEquals("camion", trayectoCamion.getTipoTransporte());
    }

    @Test
    public void testAddContenedorExitoso() {
    	trayectoCamion.addContenedor(contenedorRefrigerado);
    	assertNotNull("El contenedor debe estar asignado al muelle de destino", muelleDestino.encontrarPlazaPorContenedor(contenedorRefrigerado.getCodigo()));
    	assertTrue("El contenedor debe añadirse correctamente al trayecto", trayectoCamion.getContenedores().contains(contenedorRefrigerado));
    }

 
}
