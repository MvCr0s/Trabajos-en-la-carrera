package muelle;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import es.uva.inf.poo.maps.GPSCoordinate;
import contenedor.Contenedor;

public class MuelleTest {

    private Muelle muelle;
    private GPSCoordinate gpsCoordinate;

    @Before
    public void setUp() {
       
        gpsCoordinate = new GPSCoordinate(36.8381, -2.4597); 
        muelle = new Muelle("01", gpsCoordinate, true, 5, 3);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorNumeroDePlazasInvalido() {
        new Muelle("01", gpsCoordinate, true, 0, 3); 
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorCapacidadPorPlazaInvalida() {
        new Muelle("01", gpsCoordinate, true, 5, 0); 
    }
    
    @Test
    public void testSetId() {
        muelle.setId("02");
        assertEquals("02", muelle.getId());
    }

    @Test
    public void testSetUbicacionGPS() {
        GPSCoordinate nuevaUbicacion = new GPSCoordinate(41, 2);
        muelle.setUbicacionGPS(nuevaUbicacion);
        assertEquals(nuevaUbicacion, muelle.getUbicacionGPS());
    }
    
    @Test
    public void testSetOperativo() {
        muelle.setOperativo(false);
        assertFalse(muelle.estaOperativo());
    }
    
    @Test
    public void testSetCapacidadPlaza() {
        muelle.setCapacidadPorPlaza(5);
        assertEquals(5, muelle.getCapacidadPlaza());
    }

    @Test
    public void testGetId() {
        assertEquals("01", muelle.getId());
    }

 
    @Test
    public void testGetUbicacionGPS() {
        assertEquals(gpsCoordinate, muelle.getUbicacionGPS());
    }


    @Test
    public void testEstaOperativo() {
        assertTrue(muelle.estaOperativo());
    }


    @Test
    public void testGetNumeroDePlazas() {
        assertEquals(5, muelle.getNumeroDePlazas());
    }


    @Test
    public void testGetCapacidadPlaza() {
        assertEquals(3, muelle.getCapacidadPlaza());
    }
    


    @Test(expected = IllegalArgumentException.class)
    public void testConstructorIdInvalidoLetra() {
           new Muelle("A1", gpsCoordinate, true, 5, 3);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorIdInvalidoLongitud() {
            new Muelle("012", gpsCoordinate, true, 5, 3); 
    }

    @Test
    public void testAsignarContenedorAPlaza() {
        Contenedor contenedor = new Contenedor("CSQU3054383", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedor, 1);
        int plaza = muelle.encontrarPlazaPorContenedor("CSQU3054383");
        assertEquals(1, plaza);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAsignarContenedorAPlazaLlena() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("MAEU1234567", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor3 = new Contenedor("DEFU8901238", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);

        muelle.asignarContenedorAPlaza(contenedor1, 1);
        muelle.asignarContenedorAPlaza(contenedor2, 1);
        muelle.asignarContenedorAPlaza(contenedor3, 1);
        int nivel1 = muelle.encontrarNivelPorContenedor("CSQU3054383");
        int nivel2 = muelle.encontrarNivelPorContenedor("MAEU1234567");
        int nivel3 = muelle.encontrarNivelPorContenedor("DEFU8901238");

        assertEquals(1, nivel1);
        assertEquals(2 ,nivel2);
        assertEquals(3 ,nivel3);

        Contenedor contenedorExtra = new Contenedor("MAEU1234567", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedorExtra, 1);

    }

    @Test
    public void testEncontrarPlazaPorContenedor() {
        Contenedor contenedor = new Contenedor("CSQU3054383", 2000, 30000, 45.0, true, Contenedor.Estado.RECOGIDA);
        muelle.asignarContenedorAPlaza(contenedor, 2);
        int plazaId = muelle.encontrarPlazaPorContenedor("CSQU3054383");
        assertEquals(2, plazaId);
    }

    @Test
    public void testEncontrarNivelPorContenedor() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2500, 35000, 60.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedor1, 3);
        int nivel = muelle.encontrarNivelPorContenedor("CSQU3054383");
        assertEquals(1, nivel); 
    }

    @Test
    public void testSacarContenedorDePlaza() {
        Contenedor contenedor = new Contenedor("CSQU3054383", 2100, 31000, 55.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedor, 4);
        muelle.sacarContenedorDePlaza("CSQU3054383");
        assertNull(muelle.encontrarPlazaPorContenedor("CSQU3054383")); 

    }
    
    @Test
    public void testContarPlazasVacías() {
    	assertEquals(muelle.getNumeroDePlazas(),muelle.contarPlazasVacias());
    	assertEquals(0, muelle.contarPlazasCompletas());
    	assertEquals(0, muelle.contarPlazasSemiLlenas());
    	
    }
    
    @Test
    public void testContarPlazasConContenedores() {
        Contenedor contenedor = new Contenedor("CSQU3054383", 2100, 31000, 55.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedor, 4);
    	assertEquals(muelle.getNumeroDePlazas()-1,muelle.contarPlazasVacias());
    	assertEquals(0, muelle.contarPlazasCompletas());
    	assertEquals(1, muelle.contarPlazasSemiLlenas());
    	
    }
    
    @Test
    public void testContarPlazasCompletas() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2100, 31000, 55.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("DEFU8901238", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor3 = new Contenedor("MAEU1234567", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedor1, 4);
        muelle.asignarContenedorAPlaza(contenedor2, 4);
        muelle.asignarContenedorAPlaza(contenedor3, 4);
    	assertEquals(muelle.getNumeroDePlazas()-1,muelle.contarPlazasVacias());
    	assertEquals(1, muelle.contarPlazasCompletas());
    	assertEquals(0, muelle.contarPlazasSemiLlenas());
    	
    }
    
    @Test
    public void testSacarContenedorDePlazaAbajo() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2100, 31000, 55.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("DEFU8901238", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor3 = new Contenedor("MAEU1234567", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        boolean resultado1 = muelle.asignarContenedorAPlaza(contenedor1, 4);
        boolean resultado2 = muelle.asignarContenedorAPlaza(contenedor2, 4);
        boolean resultado3 = muelle.asignarContenedorAPlaza(contenedor3, 4);
        assertTrue(resultado1);
        assertTrue(resultado2);
        assertTrue(resultado3);
        int nivel1=muelle.encontrarNivelPorContenedor("MAEU1234567");
        int nivel2=muelle.encontrarNivelPorContenedor("DEFU8901238");
        assertEquals(3 ,nivel1);
        assertEquals(2 ,nivel2);
        muelle.sacarContenedorDePlaza("CSQU3054383");
        assertNull(muelle.encontrarPlazaPorContenedor("CSQU3054383"));
        int nivel3=muelle.encontrarNivelPorContenedor("MAEU1234567");
        int nivel4=muelle.encontrarNivelPorContenedor("DEFU8901238");
        assertEquals(2 ,nivel3);
        assertEquals(1 ,nivel4);        
    }
    
    @Test
    public void testSacarContenedorDePlazaMedio() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2100, 31000, 55.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("MAEU1234567", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor3 = new Contenedor("DEFU8901238", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedor1, 4);
        muelle.asignarContenedorAPlaza(contenedor2, 4);
        muelle.asignarContenedorAPlaza(contenedor3, 4);
        int nivel=muelle.encontrarNivelPorContenedor("DEFU8901238");
        assertEquals(3 ,nivel);
        muelle.sacarContenedorDePlaza("MAEU1234567");
        assertNull(muelle.encontrarPlazaPorContenedor("MAEU1234567"));
        int nivel1=muelle.encontrarNivelPorContenedor("DEFU8901238");
        int nivel2=muelle.encontrarNivelPorContenedor("CSQU3054383");
        assertEquals(2 ,nivel1);
        assertEquals(1 ,nivel2);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSacarContenedorYaSacado() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2100, 31000, 55.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("MAEU1234567", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedor2, 4);
        muelle.asignarContenedorAPlaza(contenedor1, 4);
        muelle.sacarContenedorDePlaza("CSQU3054383");
        assertNull(muelle.encontrarPlazaPorContenedor("CSQU3054383"));
        assertNull(muelle.encontrarNivelPorContenedor("CSQU3054383"));
        muelle.sacarContenedorDePlaza("CSQU3054383");
    }

    @Test
    public void testMuelleTieneEspacio() {
        assertTrue(muelle.tieneEspacio()); 

        String[] codigosValidos = {
        	    "CSQU3054383", "MAEU1234567", "DEFU8901238", "HLCU7788994", "CAIU9823763",
        	    "MSCU0000007", "MSCU0000012", "MSCU0000028", "MSCU0000033", "MSCU0000049",
        	    "MSCU0000054", "MSCU0000075", "MSCU0000080", "MSCU0000096", "MSCU0000115"
        	};

        int codigoIndex = 0; 

        for (int plaza = 1; plaza <= muelle.getNumeroDePlazas(); plaza++) {
            for (int nivel = 1; nivel <= muelle.getCapacidadPlaza(); nivel++) {
                if (codigoIndex < codigosValidos.length) {
                    String codigo = codigosValidos[codigoIndex++];
                    Contenedor contenedor = new Contenedor(codigo, 2000, 30000, 38.5, true, Contenedor.Estado.TRANSITO);
                    muelle.asignarContenedorAPlaza(contenedor, plaza);
                }
            }
        }

        assertFalse(muelle.tieneEspacio()); 
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testAsignarContenedorAPlazaCompleta() {
   
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("MAEU1234567", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor3 = new Contenedor("DEFU8901238", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        
        muelle.asignarContenedorAPlaza(contenedor1, 1);
        muelle.asignarContenedorAPlaza(contenedor2, 1);
        muelle.asignarContenedorAPlaza(contenedor3, 1);

        Contenedor contenedorExtra = new Contenedor("HLCU7788994", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedorExtra, 1); 
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetNumeroDePlazasCero() {
        muelle.setNumeroDePlazas(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetNumeroDePlazasNegativo() {
        muelle.setNumeroDePlazas(-5);
    }

    @Test
    public void testEqualsMuelle() {
        Muelle muelle2 = new Muelle("01", gpsCoordinate, true, 5, 3);
        assertTrue(muelle.equals(muelle2));

        Muelle muelle3 = new Muelle("02", gpsCoordinate, true, 5, 3);
        assertFalse(muelle.equals(muelle3));
    }
    
    
    @Test(expected = IllegalArgumentException.class)
    public void testColocarContenedorEncimaDeSinTecho() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2100, 31000, 55.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("MAEU1234567", 2000, 30000, 50.0, false, Contenedor.Estado.TRANSITO);
        muelle.asignarContenedorAPlaza(contenedor2, 4);
        muelle.asignarContenedorAPlaza(contenedor1, 4);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNoPermitirContenedorDuplicado() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedorDuplicado = new Contenedor("CSQU3054383", 2500, 32000, 55.0, false, Contenedor.Estado.RECOGIDA);

        muelle.asignarContenedorAPlaza(contenedor1, 1);
        muelle.asignarContenedorAPlaza(contenedorDuplicado, 2); 
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testReducirNumeroDePlazasConPlazasOcupadas() {
        Contenedor contenedor1 = new Contenedor("CSQU3054383", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("MAEU1234567", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);

        muelle.asignarContenedorAPlaza(contenedor1, 1);
        muelle.asignarContenedorAPlaza(contenedor2, 2);
        muelle.setNumeroDePlazas(1); 
    }
    
    @Test
    public void testAumentarNumeroDePlazas() {
        muelle.setNumeroDePlazas(10); 
        assertEquals(10, muelle.getNumeroDePlazas());


        Contenedor contenedor1 = new Contenedor("HLCU7788994", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("MSCU0000007", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);

        muelle.asignarContenedorAPlaza(contenedor1, 6);
        muelle.asignarContenedorAPlaza(contenedor2, 7);

        assertEquals(2, muelle.contarPlazasSemiLlenas());
    }
    
    @Test
    public void testEsPlazaCompleta() {
        Contenedor contenedor1 = new Contenedor("CAIU9823763", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor2 = new Contenedor("MSCU0000007", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);
        Contenedor contenedor3 = new Contenedor("MSCU0000012", 2000, 30000, 50.0, true, Contenedor.Estado.TRANSITO);

        muelle.asignarContenedorAPlaza(contenedor1, 1);
        muelle.asignarContenedorAPlaza(contenedor2, 1);
        muelle.asignarContenedorAPlaza(contenedor3, 1);

    
        assertTrue(muelle.contarPlazasCompletas() == 1);
    }
}
