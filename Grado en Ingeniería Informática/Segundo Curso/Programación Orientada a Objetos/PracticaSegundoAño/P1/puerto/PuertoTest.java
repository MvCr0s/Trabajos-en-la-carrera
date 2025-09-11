package puerto;

import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.Test;

import contenedor.Contenedor;

import java.util.List;
import es.uva.inf.poo.maps.GPSCoordinate;
import muelle.Muelle;
import java.util.ArrayList;

public class PuertoTest {

    private Puerto puerto;
    private Muelle muelle1;
    private Muelle muelle2;
    private GPSCoordinate gps1;
    private GPSCoordinate gps2;

    @Before
    public void setUp() {
       
        puerto = new Puerto("ES-ALM");
        gps1 = new GPSCoordinate(0, -23);  
        gps2 = new GPSCoordinate(0, -5.2);  
        muelle1 = new Muelle("01", gps1, true, 5, 5); 
        muelle2 = new Muelle("02", gps2, false, 5, 10); 
    }
    
    @Test
    public void testSetId() {
        puerto.setId("US-NYC");
        assertEquals("US-NYC", puerto.getId());
    }


    @Test
    public void testSetLocalidad() {
        puerto.setLocalidad("US-NYC");
        assertEquals("NYC", puerto.getLocalidad());
    }


    @Test
    public void testSetPais() {
        puerto.setPais("US");
        assertEquals("US", puerto.getPais());
    }
    
    @Test
    public void testGetId() {
        assertEquals("ES-ALM", puerto.getId());
    }


    @Test
    public void testGetLocalidad() {
        assertEquals("ALM", puerto.getLocalidad());
    }


    @Test
    public void testGetPais() {
        assertEquals("ES", puerto.getPais());
    }


    @Test
    public void testConstructorConCodigoValido() {
        assertEquals("ES-ALM", puerto.getId());
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConCodigoInvalidoLongitud() {
        new Puerto("ES-ALME");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConCodigoInvalidoGuion() {
        new Puerto("ESALME");
    }



    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConCodigoInvalidoMinusculas() {
        new Puerto("es-ALM");
    }

    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorConCodigoInvalidoMayusculas() {
        new Puerto("ES-alm");
    }
 
    @Test
    public void testAnadirMuelle() {
        puerto.anadirMuelle(muelle1);
        assertTrue(puerto.obtenerMuellesOperativos().contains(muelle1));
    }

  
    @Test
    public void testEliminarMuellePorId() {
        puerto.anadirMuelle(muelle1);
        puerto.eliminarMuellePorId(muelle1.getId());  
        assertFalse(puerto.obtenerMuellesOperativos().contains(muelle1));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testEliminarMuellePorIdInexistente() {
        puerto.anadirMuelle(muelle1);
        puerto.eliminarMuellePorId(muelle1.getId());  
        puerto.eliminarMuellePorId(muelle1.getId());
        
    }

   
    @Test
    public void testPuertoNoEstaCompleto() {
        puerto.anadirMuelle(muelle1); 
        assertFalse(puerto.estaCompleto()); 
    }
 
   
    @Test
    public void testPuertoEstaCompleto() {
    	String[] codigosValidos = {
    		    "CSQU3054383", "MAEU1234567", "DEFU8901238", "HLCU7788994", "CAIU9823763",
    		    "MSCU0000007", "MSCU0000012", "MSCU0000028", "MSCU0000033", "MSCU0000049",
    		    "MSCU0000054", "MSCU0000075", "MSCU0000080", "MSCU0000096", "MSCU0000115",
    		    "MSCU0000120", "MSCU0000136", "MSCU0000141", "MSCU0000157", "MSCU0000162",
    		    "MSCU0000178", "MSCU0000183", "MSCU0000199", "MSCU0000202", "MSCU0000218",
    		    "MSCU0000223", "MSCU0000239", "MSCU0000244", "MSCU0000265", "MSCU0000270",
    		    "MSCU0000286", "MSCU0000291", "MSCU0000305", "MSCU0000310", "MSCU0000326"
    		};

        puerto.anadirMuelle(muelle1);
        int codigoIndex = 0;

        for (int plaza = 1; plaza <= muelle1.getNumeroDePlazas(); plaza++) {
            for (int nivel = 1; nivel <= muelle1.getCapacidadPlaza(); nivel++) {
                if (codigoIndex < codigosValidos.length) {
                    Contenedor contenedor = new Contenedor(codigosValidos[codigoIndex++], 2000, 30000, 38.5, true, Contenedor.Estado.TRANSITO);
                    muelle1.asignarContenedorAPlaza(contenedor, plaza);
                }
            }
        }

        assertTrue(puerto.estaCompleto());
    }

    
    @Test
    public void testObtenerMuellesOperativos() {
        puerto.anadirMuelle(muelle1); 
        puerto.anadirMuelle(muelle2); 
        List<Muelle> operativos = puerto.obtenerMuellesOperativos();
        assertEquals(1, operativos.size());
        assertTrue(operativos.contains(muelle1));
    }


    @Test
    public void testObtenerMuellesConEspacio() {
        String[] codigosValidos = {
            "CSQU3054383", "MAEU1234567", "DEFU8901238", "HLCU7788994", "CAIU9823763",
            "MSCU0000007", "MSCU0000012", "MSCU0000028", "MSCU0000033", "MSCU0000049",
            "MSCU0000054", "MSCU0000075", "MSCU0000080", "MSCU0000096", "MSCU0000115",
        };

        GPSCoordinate gpsMuelle3 = new GPSCoordinate(0, -23);
        Muelle muelle3 = new Muelle("03", gpsMuelle3, true, 5, 3);
        int codigoIndex = 0;

        for (int plaza = 1; plaza <= muelle3.getNumeroDePlazas(); plaza++) {
            for (int nivel = 1; nivel <= muelle3.getCapacidadPlaza(); nivel++) {
                if (codigoIndex < codigosValidos.length) {
                    Contenedor contenedor = new Contenedor(codigosValidos[codigoIndex++], 2000, 30000, 38.5, true, Contenedor.Estado.TRANSITO);
                    muelle3.asignarContenedorAPlaza(contenedor, plaza);
                }
            }
        }

        puerto.anadirMuelle(muelle1);
        puerto.anadirMuelle(muelle2);
        puerto.anadirMuelle(muelle3);

        List<Muelle> conEspacio = puerto.obtenerMuellesConEspacio();
        assertTrue(conEspacio.contains(muelle1));
        assertTrue(conEspacio.contains(muelle2));
        assertFalse(muelle3.tieneEspacio());
        assertFalse(conEspacio.contains(muelle3));
    }


    @Test
    public void testObtenerMuellesCercanos() {
        puerto.anadirMuelle(muelle1); 
        ArrayList<Muelle> muellesCercanos = puerto.obtenerMuellesCercanos(2000, new GPSCoordinate(0, -23)); 
        assertEquals(1, muellesCercanos.size());
        assertTrue(muellesCercanos.contains(muelle1));
    }
    



    @Test(expected = IllegalArgumentException.class)
    public void testObtenerMuellesCercanosCoordenadaNula() {
        puerto.obtenerMuellesCercanos(500, null);
    }


    @Test(expected = IllegalArgumentException.class)
    public void testObtenerMuellesCercanosDistanciaNegativa() {
        puerto.obtenerMuellesCercanos(-10, gps1);
    }

 
    @Test
    public void testEqualsPuerto() {
        Puerto puerto2 = new Puerto("ES-ALM");
        assertTrue(puerto.equals(puerto2));
    }

   
    @Test
    public void testEqualsPuertoNoIgual() {
        Puerto puerto2 = new Puerto("ES-VLC");
        assertFalse(puerto.equals(puerto2));
    }
    
    
    @Test
    public void testEqualsPuertoString() {
        String puerto2 ="ES-VLC";
        assertFalse(puerto.equals(puerto2));
    }
    
}

