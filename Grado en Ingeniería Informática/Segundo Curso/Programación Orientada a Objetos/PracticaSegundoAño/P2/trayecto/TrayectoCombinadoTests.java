package trayecto;

import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.Test;

import java.time.LocalDate;
import java.util.HashSet;
import java.util.Set;

import es.uva.inf.poo.maps.GPSCoordinate;
import muelle.Muelle;
import puerto.Puerto;

public class TrayectoCombinadoTests {

    private TCamion tCamion;
    private TBarco tBarco;
    private PackCamionBarco packCamionBarco;

    @Before
    public void setUp() {
        GPSCoordinate gpsOrigen1 = new GPSCoordinate(40.0, -3.0);
        GPSCoordinate gpsDestino1 = new GPSCoordinate(41.0, -2.0);
        GPSCoordinate gpsDestino2 = new GPSCoordinate(42.0, -1.0);

        Set<String> infraestructurasAdmitidas = new HashSet<>();
        infraestructurasAdmitidas.add("barco");
        infraestructurasAdmitidas.add("camion");

        Muelle muelleOrigen1 = new Muelle("01", gpsOrigen1, true, 5, 10, infraestructurasAdmitidas);
        Muelle muelleDestino1 = new Muelle("02", gpsDestino1, true, 5, 10, infraestructurasAdmitidas);
        Muelle muelleDestino2 = new Muelle("04", gpsDestino2, true, 5, 10, infraestructurasAdmitidas);

        Puerto puertoOrigen1 = new Puerto("ES-MAD");
        String puerto="ES-ZGZ";
        Puerto puertoDestino1 = new Puerto(puerto);
        Puerto puertoOrigen2 = new Puerto(puerto);
        Puerto puertoDestino2 = new Puerto("ES-BCN");

        LocalDate fechaInicio1 = LocalDate.of(2023, 6, 1);
        LocalDate fechaFin1 = LocalDate.of(2023, 6, 3);
        LocalDate fechaInicio2 = LocalDate.of(2023, 6, 4);
        LocalDate fechaFin2 = LocalDate.of(2023, 6, 6);

        tCamion = new TCamion(muelleOrigen1, puertoOrigen1, fechaInicio1, muelleDestino1, puertoDestino1, fechaFin1);
        tBarco = new TBarco(muelleDestino1, puertoOrigen2, fechaInicio2, muelleDestino2, puertoDestino2, fechaFin2);

        packCamionBarco = new PackCamionBarco(tCamion, tBarco);
    }

    @Test
    public void testConstructorValido() {
        assertEquals(tCamion, packCamionBarco.getTrayecto1());
        assertEquals(tBarco, packCamionBarco.getTrayecto2());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorTrayectosNoConectados() {
        GPSCoordinate gpsDestino2 = new GPSCoordinate(42.0, -1.0);
        Set<String> infraestructurasAdmitidas = new HashSet<>();
        infraestructurasAdmitidas.add("barco");
        Muelle muelleOrigen1 = new Muelle("01", new GPSCoordinate(40.0, -3.0), true, 5, 10, infraestructurasAdmitidas);
        Puerto puertoDestino1 = new Puerto("ES-ZGZ");
        Puerto puertoDestino2 = new Puerto("ES-BCN");
        LocalDate fechaInicio2 = LocalDate.of(2023, 6, 4);
        LocalDate fechaFin2 = LocalDate.of(2023, 6, 6);

        TBarco barcoNoConectado = new TBarco(muelleOrigen1, puertoDestino1, fechaInicio2,
                new Muelle("04", gpsDestino2, true, 5, 10, infraestructurasAdmitidas), puertoDestino2, fechaFin2);
        new PackCamionBarco(tCamion, barcoNoConectado);
    }

    @Test
    public void testCalcularPrecioPackCamionBarco() {
        double precioBarcoOriginal = tBarco.calcularPrecioTrayecto();
        double precioBarcoConDescuento = precioBarcoOriginal * 0.85; // Descuento del 15% en el trayecto de barco
        double precioCamion = tCamion.calcularPrecioTrayecto();
        double precioEsperado = precioBarcoConDescuento + precioCamion;

        assertEquals(precioEsperado, packCamionBarco.calcularPrecioTrayecto(), 0.01);
    }

    @Test
    public void testObtenerInformacionCompleta() {
        String expectedInfo = "Trayecto combinado entre " + packCamionBarco.getPuertoOrigen().getLocalidad() + " y "
                + packCamionBarco.getPuertoDestino().getLocalidad()
                + " (PackCamionBarco)Tipo de pack: Camión + Barco (con 15% de descuento en barco)\n";
        assertEquals(expectedInfo, packCamionBarco.obtenerInformacionCompleta());
    }

    @Test
    public void testTrayectosDelPack() {
        assertEquals(tCamion, packCamionBarco.getTrayecto1());
        assertEquals(tBarco, packCamionBarco.getTrayecto2());
    }
}
