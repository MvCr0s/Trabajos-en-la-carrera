package muelle;


import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import es.uva.inf.poo.maps.GPSCoordinate;

import contenedor.Contenedor;
import contenedor.ContenedorFlatRack;

/**
 * Representa un muelle que gestiona la ubicación y apilamiento de contenedores en diferentes plazas.
 * Permite asignar y retirar contenedores, verificando compatibilidad con tipos de transporte admitidos.
 * @author alfdelv
 * @author mardedi
 */
public class Muelle {
    private static final Set<String> INFRAESTRUCTURAS_VALIDAS = Set.of("barco", "tren", "camion");

    private String id;
    private GPSCoordinate ubicacionGPS;
    private boolean operativo;
    private int plazas;
    private int capacidadPorPlaza;
    private Map<Integer, Contenedor[]> plazasDeContenedores;
    private Set<String> infraestructurasAdmitidas; 

    /**
     * Constructor de la clase Muelle.
     *
     * @param id identificador del muelle en formato de dos dígitos.
     * @param ubicacionGPS coordenadas GPS del muelle.
     * @param operativo estado operativo del muelle.
     * @param plazas número total de plazas disponibles en el muelle.
     * @param capacidadPorPlaza capacidad máxima de contenedores por cada plaza.
     * @param infraestructurasAdmitidas conjunto de infraestructuras admitidas (barco, tren, camion).
     * @throws IllegalArgumentException si el id no cumple con el formato de dos dígitos o si el número de plazas es menor o igual a cero.
     */
    public Muelle(String id, GPSCoordinate ubicacionGPS, boolean operativo, int plazas, int capacidadPorPlaza, Set<String> infraestructurasAdmitidas) {
        this.plazasDeContenedores = new HashMap<>();
        setNumeroDePlazas(plazas);
        setCapacidadPorPlaza(capacidadPorPlaza);
        setId(id);
        setUbicacionGPS(ubicacionGPS);
        setOperativo(operativo);
        setInfraestructurasAdmitidas(infraestructurasAdmitidas);
        setPlazasDeContenedores();
    }
    
    /**
     * Verifica si el identificador proporcionado es válido.
     *
     * @param id Identificador a verificar.
     * @return true si el id es válido, false en caso contrario.
     */  
    public boolean esIdentificadorValido(String id) {
        if (id.length() != 2) {
            return false;
        }
        for (int i = 0; i < 2; i++) {
            if (!Character.isDigit(id.charAt(i))) {
                return false; 
            }
        }
        return true; 
    }
    
    
    /**
     * Obtiene el identificador del muelle.
     *
     * @return Identificador del muelle.
     */
    public String getId() {
        return id;    
    }
    
    
    /**
     * Asigna un nuevo identificador al muelle.
     *
     * @param id Nuevo identificador del muelle.
     * @throws IllegalArgumentException si el id no cumple con el formato de dos dígitos o si el número de plazas es menor o igual a cero.
     */
   public void setId(String id) {
	   
	   if (!esIdentificadorValido(id)) {
           throw new IllegalArgumentException("Código inválido. Debe seguir el formato 00");
       }
	   else {
		   this.id=id;
	   }
   }
   
   
   /**
    * Obtiene la ubicación GPS del muelle.
    *
    * @return Coordenadas GPS del muelle.
    */
   public GPSCoordinate getUbicacionGPS() {
       return ubicacionGPS;
   }
   
   
   /**
    * Asigna una nueva ubicación GPS al muelle.
    *
    * @param ubicacionGPS Nueva coordenada GPS.
    */
   public void setUbicacionGPS(GPSCoordinate ubicacionGPS) {
       this.ubicacionGPS = ubicacionGPS;
   }
   

   /**
    * Verifica si el muelle está operativo.
    *
    * @return true si el muelle está operativo, false en caso contrario.
    */
   public boolean estaOperativo() {
       return operativo;
   }
   
   
   /**
    * Establece el estado operativo del muelle.
    *
    * @param operativo Nuevo estado operativo del muelle.
    */
   public void setOperativo(boolean operativo) {
       this.operativo = operativo;
   }
   
   
   /**
    * Establece el número total de plazas en el muelle.
    *
    * @param plazas Número total de plazas.
    * @throws IllegalArgumentException si el número de plazas es menor o igual a cero.
    */
   public void setNumeroDePlazas(int plazas) {
	    if (plazas <= 0) {
	        throw new IllegalArgumentException("El número de plazas debe ser mayor que cero.");
	    }
	    if (plazas < this.plazasDeContenedores.size()) {
	        throw new IllegalArgumentException("No se puede reducir el número de plazas por debajo del número de contenedores actuales.");
	    }
	    if (plazas > this.plazas) {
	        for (int i = this.plazas + 1; i <= plazas; i++) {
	            plazasDeContenedores.put(i, new Contenedor[capacidadPorPlaza]);
	        }
	    }
	    this.plazas = plazas;
	}
   
   
   /**
    * Obtiene el número total de plazas en el muelle.
    *
    * @return Número de plazas.
    */
   public int getNumeroDePlazas() {
       return plazas;
   }
   
   
   /**
    * Establece la capacidad máxima por plaza.
    *
    * @param capacidadPorPlaza Capacidad máxima de contenedores por plaza.
    * @throws IllegalArgumentException Si la capacidad es incorrecta.
    */
   public void setCapacidadPorPlaza(int capacidadPorPlaza) {
       if (capacidadPorPlaza <= 0) {
           throw new IllegalArgumentException("La capacidad por plaza debe ser mayor que cero.");
       }
       this.capacidadPorPlaza = capacidadPorPlaza;
   }
   
   
   /**
    * Obtiene la capacidad máxima de contenedores en una plaza.
    *
    * @return Capacidad por plaza.
    */
   public int getCapacidadPlaza() {
       return capacidadPorPlaza;
   }
   
   
   /**
    * Inicializa el mapa de plazas de contenedores vacías en el muelle.
    */
   private void setPlazasDeContenedores() {
	    this.plazasDeContenedores.clear();
	    for (int i = 1; i <= plazas; i++) {
	        plazasDeContenedores.put(i, new Contenedor[capacidadPorPlaza]);
	    }
	}
   
   
   /**
    * Verifica si una plaza está vacía.
    *
    * @param apilados Array de contenedores en la plaza.
    * @return true si la plaza está vacía, false en caso contrario.
    */
   private boolean esPlazaVacia(Contenedor[] apilados) {
       return apilados[0] == null;
   }
   
   
   /**
    * Verifica si una plaza está completa.
    *
    * @param apilados Array de contenedores en la plaza.
    * @return true si la plaza está completa, false en caso contrario.
    */
   private boolean esPlazaCompleta(Contenedor[] apilados) {
       return apilados[apilados.length - 1] != null;
   }
   
   
   /**
    * Cuenta el número de plazas vacías en el muelle.
    *
    * @return Número de plazas vacías.
    */
   public int contarPlazasVacias() {
       int plazasVacias = 0;
       for (Contenedor[] apilados : plazasDeContenedores.values()) {
           if (esPlazaVacia(apilados)) {
               plazasVacias++;
           }
       }
       return plazasVacias;
   }
   
   
   /**
    * Cuenta el número de plazas completas en el muelle.
    *
    * @return Número de plazas completas.
    */
   public int contarPlazasCompletas() {
       int plazasCompletas = 0;
       for (Contenedor[] apilados : plazasDeContenedores.values()) {
           if (esPlazaCompleta(apilados)) {
               plazasCompletas++;
           }
       }
       return plazasCompletas;
   }

   
   /**
    * Cuenta el número de plazas semi-llenas en el muelle.
    *
    * @return Número de plazas semi-llenas.
    */
   public int contarPlazasSemiLlenas() {
       int plazasSemiLlenas = 0;
       for (Contenedor[] apilados : plazasDeContenedores.values()) {
           if (!esPlazaVacia(apilados) && !esPlazaCompleta(apilados)) {
               plazasSemiLlenas++;
           }
       }
       return plazasSemiLlenas;
   }
   

   /**
    * Encuentra la plaza en la que está ubicado un contenedor dado su código.
    *
    * @param codigoContenedor Código del contenedor a buscar.
    * @return El número de la plaza si se encuentra, null en caso contrario.
    */
    public Integer encontrarPlazaPorContenedor(String codigoContenedor) {
        for (Map.Entry<Integer, Contenedor[]> entry : plazasDeContenedores.entrySet()) {
            Contenedor[] apilados = entry.getValue();
            for (Contenedor contenedor : apilados) {
                if (contenedor != null && contenedor.getCodigo().equals(codigoContenedor)) {
                    return entry.getKey();
                }
            }
        }
        return null; 
    }
    
    
    /**
     * Encuentra el nivel de apilamiento de un contenedor en su plaza dado su código.
     *
     * @param codigoContenedor Código del contenedor a buscar.
     * @return Nivel de apilamiento si se encuentra, null en caso contrario.
     */   
    public Integer encontrarNivelPorContenedor(String codigoContenedor) {
        for (Contenedor[] apilados : plazasDeContenedores.values()) {
            for (int nivel = 0; nivel < apilados.length; nivel++) {
                if (apilados[nivel] != null && apilados[nivel].getCodigo().equals(codigoContenedor)) {
                    return nivel + 1;
                }
            }
        }
        return null; 
    }
    
    
    /**
     * Establece las infraestructuras admitidas por el muelle.
     *
     * @param infraestructuras conjunto de infraestructuras admitidas.
     * @throws IllegalArgumentException si el conjunto de infraestructuras está vacío.
     */
    public void setInfraestructurasAdmitidas(Set<String> infraestructuras) {
        if (infraestructuras == null || infraestructuras.isEmpty()) {
            throw new IllegalArgumentException("El conjunto de infraestructuras no puede estar vacío.");
        }
        for (String infraestructura : infraestructuras) {
            if (!INFRAESTRUCTURAS_VALIDAS.contains(infraestructura)) {
                throw new IllegalArgumentException("Infraestructura no válida. ");
            }
        }
        this.infraestructurasAdmitidas = infraestructuras;
    }

    /**
     * Obtiene las infraestructuras admitidas por el muelle.
     *
     * @return conjunto de infraestructuras admitidas.
     */
    public Set<String> getInfraestructurasAdmitidas() {
        return infraestructurasAdmitidas;
    }
    
    
    /**
     * Verifica si un contenedor es compatible con las infraestructuras admitidas del muelle.
     *
     * @param contenedor el contenedor a verificar.
     * @return true si el contenedor es compatible, false en caso contrario.
     */
    public boolean esCompatibleConInfraestructuras(Contenedor contenedor) {
        for (String infraestructura : infraestructurasAdmitidas) {
            if (contenedor.esCompatibleConInfraestructura(infraestructura)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Asigna un contenedor a una plaza en el muelle.
     * Verifica si el contenedor es compatible con las infraestructuras del muelle, si no ha sido
     * previamente asignado, y luego intenta asignarlo a una plaza. En el caso de los contenedores
     * de tipo {@link ContenedorFlatRack}, se requiere asignar dos plazas consecutivas disponibles.
     *
     * @param contenedor El contenedor que se desea asignar.
     * @param plazaId    El identificador de la plaza donde se desea asignar el contenedor.
     * @return {@code true} si el contenedor fue asignado con éxito; {@code false} si no se cumplen 
     *         las condiciones de asignación (plazas llenas, incompatibilidad de estructura, etc.).
     * @throws IllegalArgumentException Si el contenedor no es compatible con las infraestructuras admitidas 
     *                                  o si ya está asignado a una plaza en este muelle.
     */
    public boolean asignarContenedorAPlaza(Contenedor contenedor, int plazaId) {
        validarCompatibilidad(contenedor);
        validarAsignacionExistente(contenedor);

        boolean asignado;

        if (contenedor instanceof ContenedorFlatRack) {
            asignado = asignarContenedorFlatRack((ContenedorFlatRack) contenedor, plazaId);
        } else {
            asignado = asignarContenedorNormal(contenedor, plazaId);
        }

        return asignado; 
    }

    private void validarCompatibilidad(Contenedor contenedor) {
        if (!esCompatibleConInfraestructuras(contenedor)) {
            throw new IllegalArgumentException("El contenedor no es compatible con las infraestructuras admitidas por el muelle.");
        }
    }

    private void validarAsignacionExistente(Contenedor contenedor) {
        if (encontrarPlazaPorContenedor(contenedor.getCodigo()) != null) {
            throw new IllegalArgumentException("El contenedor ya está asignado a una plaza en este muelle.");
        }
    }

    private boolean asignarContenedorFlatRack(ContenedorFlatRack flatRack, int plazaId) {
        if (plazaId  > plazas || !esPlazaVacia(plazasDeContenedores.get(plazaId)) || !esPlazaVacia(plazasDeContenedores.get(plazaId + 1))) {
            return false; 
        }
        plazasDeContenedores.get(plazaId)[0] = flatRack;
        plazasDeContenedores.get(plazaId + 1)[0] = flatRack;
        return true;
    }

    private boolean asignarContenedorNormal(Contenedor contenedor, int plazaId) {
        Contenedor[] plaza = plazasDeContenedores.get(plazaId);


        if (esPlazaCompleta(plaza)) {
            throw new IllegalArgumentException("La plaza está completa y no puede asignarse más contenedores.");
        }

        for (int i = 0; i < plaza.length; i++) {
            if (plaza[i] == null) {
                if (i > 0) { 
                    Contenedor contenedorDebajo = plaza[i - 1];
                    if (!contenedorDebajo.getTecho()) {
                        throw new IllegalArgumentException("No se puede colocar un contenedor encima de otro sin techo.");
                    }
                }
                plaza[i] = contenedor;
                return true; 
            }
        }

        return false;
    }

    /**
     * Retira un contenedor de su plaza en el muelle.
     *
     * @param codigoContenedor Código del contenedor a retirar.
     * @throws IllegalArgumentException si el contenedor no se encuentra en ninguna plaza.
     */
    public void sacarContenedorDePlaza(String codigoContenedor) {
        boolean encontrado = false; 

        for (Contenedor[] apilados : plazasDeContenedores.values()) {
        
            for (int nivel = apilados.length - 1; nivel >= 0; nivel--) {   
            	
                if (apilados[nivel] != null && apilados[nivel].getCodigo().equals(codigoContenedor)) {
                    apilados[nivel] = null; 
                    encontrado = true; 

                    for (int i = nivel; i < apilados.length - 1; i++) {
                        apilados[i] = apilados[i + 1];  
                        apilados[i + 1] = null; 
                    }
                }
            }
        }
        if (!encontrado) {
            throw new IllegalArgumentException("El contenedor no se encuentra en ninguna plaza.");
        }
    }
    
    
    /**
     * Verifica si el muelle tiene espacio disponible en alguna de sus plazas.
     *
     * @return true si hay espacio disponible, false en caso contrario.
     */
    public boolean tieneEspacio() {
    	  
        for (Contenedor[] apilados : plazasDeContenedores.values()) {
        
            if (!esPlazaCompleta(apilados)) {
                return true;  
            }
        }
        return false;  
    }
    
    
    
    
    /**
     * Compara este muelle con otro objeto para verificar si son iguales.
     *
     * @param obj El objeto a comparar.
     * @return true si los muelles son iguales (tienen el mismo id y ubicación GPS), false en caso contrario.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        Muelle muelle = (Muelle) obj;
        return this.id.equals(muelle.id) && this.ubicacionGPS.equals(muelle.ubicacionGPS);  
    }

    /**
     * Devuelve el código hash de este muelle.
     * 
     * @return Un entero que representa el código hash del muelle.
     */
    @Override
    public int hashCode() {
        return Objects.hash(id, ubicacionGPS);
    }
}
