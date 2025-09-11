package trayecto;

import java.time.LocalDate;
import es.uva.inf.poo.maps.GPSCoordinate;
import puerto.Puerto;
import muelle.Muelle;
import java.util.ArrayList; 
import java.util.List;
import contenedor.Contenedor;

/**
 * Representa un trayecto entre dos puertos a través de muelles y calcula su precio.
 * Esta es una clase abstracta que puede ser extendida por diferentes tipos de trayectos, 
 * como trayectos simples o combinados, que calculan el precio de acuerdo con distintos
 * criterios de costes (por día, por kilómetro, etc.).
 * @author alfdelv
 * @author mardedi
 */
public abstract class Trayecto {
    private Muelle muelleOrigen;
    private Puerto puertoOrigen;
    private LocalDate fechaInicio;
    private Muelle muelleDestino;
    private Puerto puertoDestino;
    private LocalDate fechaFin;
    private String tipoTransporte; 
    private List<Contenedor> contenedores; 


    /**
     * Crea un trayecto entre dos puertos con sus muelles y fechas.
     * 
     * @param muelleOrigen Muelle de origen
     * @param puertoOrigen Puerto de origen
     * @param fechaInicio Fecha de inicio del trayecto
     * @param muelleDestino Muelle de destino
     * @param puertoDestino Puerto de destino
     * @param fechaFin Fecha de fin del trayecto
     * @param tipoTransporte Tipo de transporte utilizado en el trayecto
     */
    protected Trayecto(Muelle muelleOrigen, Puerto puertoOrigen, LocalDate fechaInicio, 
                    Muelle muelleDestino, Puerto puertoDestino, LocalDate fechaFin, String tipoTransporte) { 
        this.muelleOrigen = muelleOrigen;
        this.puertoOrigen = puertoOrigen;
        this.fechaInicio = fechaInicio;
        this.muelleDestino = muelleDestino;
        this.puertoDestino = puertoDestino;
        this.fechaFin = fechaFin;
        this.tipoTransporte = tipoTransporte; 
        this.contenedores = new ArrayList<>();
    }

    /**
     * Método abstracto para calcular el precio del trayecto. 
     * Este método debe ser implementado por las clases derivadas 
     * que especifiquen la lógica de cálculo del precio para diferentes 
     * tipos de trayecto (barco, tren, camión, etc.).
     * 
     * @return El precio del trayecto
     */
    public abstract double calcularPrecioTrayecto();

    /**
     * Método abstracto para obtener la información completa del trayecto,
     * incluyendo los detalles de los puertos, muelles y fechas involucradas.
     * 
     * @return Información detallada del trayecto
     */
    public abstract String obtenerInformacionCompleta();


    /**
     * Obtiene el muelle de origen del trayecto.
     * 
     * @return El muelle de origen
     */
    public Muelle getMuelleOrigen() {
        return muelleOrigen;
    }

    /**
     * Establece el muelle de origen del trayecto.
     * 
     * @param muelleOrigen El muelle de origen
     */
    public void setMuelleOrigen(Muelle muelleOrigen) {
        this.muelleOrigen = muelleOrigen;
    }

    /**
     * Obtiene el puerto de origen del trayecto.
     * 
     * @return El puerto de origen
     */
    public Puerto getPuertoOrigen() {
        return puertoOrigen;
    }

    /**
     * Establece el puerto de origen del trayecto.
     * 
     * @param puertoOrigen El puerto de origen
     */
    public void setPuertoOrigen(Puerto puertoOrigen) {
        this.puertoOrigen = puertoOrigen;
    }

    /**
     * Obtiene la fecha de inicio del trayecto.
     * 
     * @return La fecha de inicio del trayecto
     */
    public LocalDate getFechaInicio() {
        return fechaInicio;
    }

    /**
     * Establece la fecha de inicio del trayecto.
     * 
     * @param fechaInicio La fecha de inicio del trayecto
     */
    public void setFechaInicio(LocalDate fechaInicio) {
        this.fechaInicio = fechaInicio;
    }

    /**
     * Obtiene el muelle de destino del trayecto.
     * 
     * @return El muelle de destino
     */
    public Muelle getMuelleDestino() {
        return muelleDestino;
    }

    /**
     * Establece el muelle de destino del trayecto.
     * 
     * @param muelleDestino El muelle de destino
     */
    public void setMuelleDestino(Muelle muelleDestino) {
    	if(muelleDestino.equals(muelleOrigen)) {
    		throw new IllegalArgumentException("El muelle destino no puede ser el mismo q el de origen");
    	}
        this.muelleDestino = muelleDestino;
    }

    /**
     * Obtiene el puerto de destino del trayecto.
     * 
     * @return El puerto de destino
     */
    public Puerto getPuertoDestino() {
        return puertoDestino;
    }

    /**
     * Establece el puerto de destino del trayecto.
     * 
     * @param puertoDestino El puerto de destino
     */
    public void setPuertoDestino(Puerto puertoDestino) {
    	if(puertoDestino.equals(puertoOrigen)) {
    		throw new IllegalArgumentException("El puerto destino no piuede ser el mismo q el de origen");
    	}
        this.puertoDestino = puertoDestino;
    }

    /**
     * Obtiene la fecha de fin del trayecto.
     * 
     * @return La fecha de fin del trayecto
     */
    public LocalDate getFechaFin() {
        return fechaFin;
    }

    /**
     * Establece la fecha de fin del trayecto.
     * 
     * @param fechaFin La fecha de fin del trayecto
     * @throws IllegalArgumentException Si la fecha de fin es anterior a la fecha de inicio
     */
    public void setFechaFin(LocalDate fechaFin) {
    	if (fechaFin.isAfter(fechaInicio)) {
            this.fechaFin = fechaFin;
        } else {
            throw new IllegalArgumentException("La fecha de llegada es anterior a la fecha de salida");
        }
    }

    /**
     * Calcula la distancia en millas marinas entre el muelle de origen y el muelle de destino.
     * Utiliza las coordenadas GPS de los muelles para calcular la distancia.
     * 
     * @return La distancia entre los muelles en millas marinas
     */
    public double obtenerDistanciaMillasMarinas() {
        GPSCoordinate coordenadasOrigen = muelleOrigen.getUbicacionGPS();
        GPSCoordinate coordenadasDestino = muelleDestino.getUbicacionGPS();

        double distanciaKilometros = coordenadasOrigen.getDistanceTo(coordenadasDestino);
        return distanciaKilometros / 1.60934;  
    }
    
    /**
     * Añade un contenedor al trayecto, retirándolo del muelle de origen 
     * y asignándolo al muelle de destino.
     * 
     * @param contenedor Contenedor a añadir.
     */
    public void addContenedor(Contenedor contenedor) {
        for (int i = 1; i <= muelleDestino.getNumeroDePlazas(); i++) {
            if (muelleDestino.asignarContenedorAPlaza(contenedor, i)) {
                contenedores.add(contenedor);
                muelleOrigen.sacarContenedorDePlaza(contenedor.getCodigo());
                return;
            }
        }
    }
    
    /**
     * Obtiene el tipo de transporte del trayecto.
     * 
     * @return Tipo de transporte.
     */
    public String getTipoTransporte() { 
        return tipoTransporte;
    }
    
    
    /**
     * Devuelve una lista inmutable con los contenedores asociados al trayecto.
     * 
     * @return Lista de contenedores asignados al trayecto.
     */
    public List<Contenedor> getContenedores() {
        return List.copyOf(contenedores);
    }

    
}
