package trayecto;

import static org.junit.Assert.*;


import org.junit.Test;

import java.time.LocalDate;
import java.util.HashSet;
import java.util.Set;
import org.junit.Before;
import es.uva.inf.poo.maps.GPSCoordinate;
import muelle.Muelle;
import puerto.Puerto;

public class TrayectoSimpleTest {


    private TCamion trayectoCamion;
    private TBarco trayectoBarco;
    private Set<String> infraestructurasAdmitidas = new HashSet<>();

    @Before
    public void setUp() {
        GPSCoordinate gpsOrigen = new GPSCoordinate(38.0, -1.0);
        GPSCoordinate gpsDestino = new GPSCoordinate(39.5, 3.0);
        infraestructurasAdmitidas.add("barco");
        infraestructurasAdmitidas.add("tren");
        infraestructurasAdmitidas.add("camion");

        Muelle muelleOrigen = new Muelle("01", gpsOrigen, true, 5, 10, infraestructurasAdmitidas);
        Muelle muelleDestino = new Muelle("02", gpsDestino, true, 5, 10, infraestructurasAdmitidas);

        Puerto puertoOrigen = new Puerto("ES-MUR");
        Puerto puertoDestino = new Puerto("ES-PMI");

        LocalDate fechaInicio = LocalDate.of(2023, 6, 10);
        LocalDate fechaFin = LocalDate.of(2023, 6, 15);

        trayectoCamion = new TCamion(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin);
        trayectoBarco = new TBarco(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin);
    }


    @Test
    public void testCalcularPrecioBarco() {
    	double precio = 20000.0;
        assertEquals(trayectoBarco.calcularPrecioTrayecto(), precio, 0.01);
    }

    @Test
    public void testObtenerInformacionCompleta() {
        String infoEsperada = "Trayecto simple desde " + trayectoCamion.getPuertoOrigen().getLocalidad() + " hasta " + trayectoCamion.getPuertoDestino().getLocalidad()  + " (TCamion)";
        assertEquals(infoEsperada, trayectoCamion.obtenerInformacionCompleta());
    }

    
}