package puerto;

import java.util.ArrayList;

import java.util.List;
import es.uva.inf.poo.maps.GPSCoordinate;

import muelle.Muelle;


/**
 * Clase que representa un puerto marítimo, con varios muelles para almacenar contenedores.
 * Cada puerto tiene un identificador único, localidad, país, y una lista de muelles.
 * @author alfdelv
 * @author mardedi
 */
public class Puerto {
    private String id;  
    private String localidad;
    private String pais;
    private List<Muelle> muelles;
    

    /**
     * Constructor para crear un nuevo puerto a partir de un identificador.
     * El identificador debe tener el formato "XX-YYY", donde:
     * XX son las dos letras del país en mayúsculas.
     * YYY son tres letras que representan la localidad en mayúsculas.
     * @param id Identificador del puerto.
     * @throws IllegalArgumentException si el identificador no sigue el formato especificado o es nulo.
     */
    public Puerto(String id) {
        setId(id);
        setPais(id);
        setLocalidad(id);
        this.muelles = new ArrayList<>();
    }
    
    
    /**
     * Verifica si el identificador del puerto es válido (formato "XX-YYY").
     *
     * @param id Identificador a validar.
     * @return true si el identificador es válido, false en caso contrario.
     */
    private boolean esCodigoValido(String id) {
        if (id.length() != 6) {
            return false;
        }
        for (int i = 0; i < 2; i++) {
            if (!Character.isUpperCase(id.charAt(i))) {
                return false;  
            }
        }       
        if (id.charAt(2) != '-') {
            return false;
        }
        for (int i = 3; i < 6; i++) {
            if (!Character.isUpperCase(id.charAt(i))) {
                return false; 
            }
        }
        return true;  
    }
    
  
    /**
     * Obtiene el identificador del puerto.
     *
     * @return El identificador del puerto.
     */
    public String getId() {
        return id;
    }
    
    
    /**
     * Establece el identificador del puerto.
     *
     * @param id Identificador del puerto.
     * @throws IllegalArgumentException si el identificador no sigue el formato "XX-YYY".
     */
    public void setId(String id) {
    	if (!esCodigoValido(id)) {
            throw new IllegalArgumentException("C�digo inv�lido. Debe seguir el formato XX-YYY.");
        }
    	this.id=id;
    }
    
    
    /**
     * Obtiene la localidad del puerto.
     *
     * @return La localidad del puerto.
     */
    public String getLocalidad() {
        return localidad;
    }
    
    
    /**
     * Establece la localidad del puerto a partir del identificador.
     *
     * @param id Identificador del puerto, que debe seguir el formato "XX-YYY".
     */
    public void setLocalidad(String id) {
    	this.localidad=id.substring(3);
    }
    
    
    /**
     * Obtiene el país del puerto.
     *
     * @return El país del puerto.
     */
    public String getPais() {
        return pais;
    }
    
    
    /**
     * Establece el país del puerto a partir del identificador.
     *
     * @param id Identificador del puerto, que debe seguir el formato "XX-YYY".
     */
    public void setPais(String id) {
    	this.pais = id.substring(0, 2);
    }
    
    
    /**
     * Añade un nuevo muelle al puerto.
     *
     * @param nuevoMuelle Muelle a añadir.
     */
    public void anadirMuelle(Muelle nuevoMuelle) {
        muelles.add(nuevoMuelle);
    }
    

    /**
     * Elimina un muelle del puerto por su identificador.
     *
     * @param idMuelle Identificador del muelle a eliminar.
     * @throws IllegalArgumentException si no se encuentra el muelle con el identificador dado.
     */
    public void eliminarMuellePorId(String idMuelle) {
        for (Muelle muelle : new ArrayList<>(muelles)) { 
            if (muelle.getId().equals(idMuelle)) {
                muelles.remove(muelle);
                return;
            }
        }
        throw new IllegalArgumentException("No se encontró el muelle");
    }
    

    /**
     * Verifica si el puerto está completo, es decir, si todos los muelles están llenos.
     *
     * @return true si todos los muelles están llenos, false si al menos un muelle tiene espacio disponible.
     */
    public boolean estaCompleto() {
    	boolean completo=true;
        for (Muelle muelle : muelles) {
            if (muelle.tieneEspacio()) {
                completo=false;  
            }
        }
        return completo;  
    }

    
    /**
     * Obtiene una lista de muelles operativos del puerto.
     *
     * @return Lista de muelles operativos.
     */
    public List<Muelle> obtenerMuellesOperativos() {
        List<Muelle> muellesOperativos = new ArrayList<>();
        for (Muelle muelle : muelles) {
            if (muelle.estaOperativo()) {
                muellesOperativos.add(muelle); 
            }
        }
        return muellesOperativos;
    }

    
    /**
     * Obtiene una lista de muelles que tienen espacio disponible.
     *
     * @return Lista de muelles con espacio disponible.
     */
    public List<Muelle> obtenerMuellesConEspacio() {
        List<Muelle> muellesConEspacio = new ArrayList<>();
        for (Muelle muelle : muelles) {
            if (muelle.tieneEspacio()) {
                muellesConEspacio.add(muelle);
            }
        }
        return muellesConEspacio;
    }
    

    /**
     * Obtiene una lista de muelles cercanos a un punto GPS dentro de una distancia máxima dada.
     *
     * @param distanciaMaxima Distancia máxima en kilómetros.
     * @param puntoGPS Coordenadas GPS de referencia.
     * @return Lista de muelles cercanos al punto GPS.
     * @throws IllegalArgumentException si el punto GPS es nulo o la distancia máxima no es positiva.
     */
    public List<Muelle> obtenerMuellesCercanos(double distanciaMaxima, GPSCoordinate puntoGPS) {
        if (puntoGPS == null) {
            throw new IllegalArgumentException("Las coordenadas no pueden ser nulas");
        }
        if (distanciaMaxima <= 0) {
            throw new IllegalArgumentException("La distancia debe ser positiva");
        }

        ArrayList<Muelle> muellesCercanos = new ArrayList<>();
        for (Muelle muelle : muelles) { 
            double distancia = (puntoGPS).getDistanceTo(muelle.getUbicacionGPS());
            if (distancia <= distanciaMaxima) {
                muellesCercanos.add(muelle);
            }
        }
        return muellesCercanos;
    }
    
    
    /**
     * Compara este puerto con otro objeto para verificar si son iguales.
     *
     * @param obj Objeto a comparar.
     * @return true si el objeto es un puerto con el mismo identificador, false en caso contrario.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        Puerto puerto = (Puerto) obj;
        return this.id.equals(puerto.id);  
    }  
    
    @Override
    public int hashCode() {
        return id.hashCode();
    }
}
