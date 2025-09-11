package trayecto;

import static org.junit.Assert.*;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import es.uva.inf.poo.maps.GPSCoordinate;
import muelle.Muelle;
import puerto.Puerto;

public class PackCamionBarcoTest {

    private Muelle muelleOrigen1;
    private Muelle muelleDestino1;
    private Muelle muelleOrigen2;
    private Muelle muelleDestino2;

    private Puerto puertoOrigen1;
    private Puerto puertoDestino1;
    private Puerto puertoOrigen2;
    private Puerto puertoDestino2;

    private LocalDate fechaInicio1;
    private LocalDate fechaFin1;
    private LocalDate fechaInicio2;
    private LocalDate fechaFin2;

    private TCamion tCamion;
    private TBarco tBarco;
    private PackCamionBarco packCamionBarco;

    @Before
    public void setUp() {
        GPSCoordinate gpsOrigen1 = new GPSCoordinate(40.0, -3.0);
        GPSCoordinate gpsDestino1 = new GPSCoordinate(41.0, -2.0);
        GPSCoordinate gpsOrigen2 = new GPSCoordinate(41.0, -2.0);
        GPSCoordinate gpsDestino2 = new GPSCoordinate(42.0, -1.0);

        Set<String> infraestructurasAdmitidas = new HashSet<>();
        infraestructurasAdmitidas.add("barco");
        infraestructurasAdmitidas.add("camion");

        muelleOrigen1 = new Muelle("01", gpsOrigen1, true, 5, 10, infraestructurasAdmitidas);
        muelleDestino1 = new Muelle("02", gpsDestino1, true, 5, 10, infraestructurasAdmitidas);
        muelleDestino2 = new Muelle("04", gpsDestino2, true, 5, 10, infraestructurasAdmitidas);

        puertoOrigen1 = new Puerto("ES-MAD");
        puertoDestino1 = new Puerto("ES-ZGZ");
        puertoOrigen2 = new Puerto("ES-ZGZ");
        puertoDestino2 = new Puerto("ES-BCN");

        fechaInicio1 = LocalDate.of(2023, 6, 1);
        fechaFin1 = LocalDate.of(2023, 6, 3);
        fechaInicio2 = LocalDate.of(2023, 6, 4);
        fechaFin2 = LocalDate.of(2023, 6, 6);

        tCamion = new TCamion(muelleOrigen1, puertoOrigen1, fechaInicio1, muelleDestino1, puertoDestino1, fechaFin1);
        tBarco = new TBarco(muelleDestino1, puertoOrigen2, fechaInicio2, muelleDestino2, puertoDestino2, fechaFin2);

        packCamionBarco = new PackCamionBarco(tCamion, tBarco);
    }

    @Test
    public void testConstructorValido() {
        assertNotNull(packCamionBarco);
        assertEquals(tCamion, packCamionBarco.getTrayecto1());
        assertEquals(tBarco, packCamionBarco.getTrayecto2());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConTrayectosInvalidos() {
        TBarco barcoNoConectado = new TBarco(muelleOrigen1, puertoDestino1, fechaInicio2, muelleDestino2, puertoDestino2, fechaFin2);
        new PackCamionBarco(tCamion, barcoNoConectado);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConTiposIncorrectos() {
        TBarco barcoInvalido = new TBarco(muelleOrigen1, puertoOrigen1, fechaInicio1, muelleDestino1, puertoDestino1, fechaFin1);
        new PackCamionBarco(barcoInvalido, tBarco);
    }

    @Test
    public void testCalcularPrecio() {
        double precioCamion = tCamion.calcularPrecioTrayecto();
        double precioBarcoOriginal = tBarco.calcularPrecioTrayecto();
        double precioBarcoConDescuento = precioBarcoOriginal * 0.85; // 15% descuento en barco

        double precioEsperado = precioCamion + precioBarcoConDescuento;

        assertEquals(precioEsperado, packCamionBarco.calcularPrecioTrayecto(), 0.01);
    }

    @Test
    public void testObtenerInformacionCompleta() {
        String infoEsperada = "Trayecto combinado entre MAD y BCN (PackCamionBarco)Tipo de pack: Camión + Barco (con 15% de descuento en barco)\n";
        assertEquals(infoEsperada, packCamionBarco.obtenerInformacionCompleta());
    }

    @Test
    public void testTrayectosDelPack() {
        assertEquals(tCamion, packCamionBarco.getTrayecto1());
        assertEquals(tBarco, packCamionBarco.getTrayecto2());
    }

}