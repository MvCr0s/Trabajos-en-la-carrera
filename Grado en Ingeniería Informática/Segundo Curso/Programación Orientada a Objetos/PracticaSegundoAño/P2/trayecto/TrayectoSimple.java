package trayecto;

import java.time.LocalDate;
import muelle.Muelle;
import puerto.Puerto;

/**
 * Representa un trayecto simple entre dos puertos a través de muelles.
 * Esta clase sirve como base para los trayectos simples como barco, tren y camión.
 * @author alfdelv
 * @author mardedi
 */
public abstract class TrayectoSimple extends Trayecto {

    /**
     * Constructor de un trayecto simple.
     * 
     * @param muelleOrigen Muelle de origen
     * @param puertoOrigen Puerto de origen
     * @param fechaInicio Fecha de inicio
     * @param muelleDestino Muelle de destino
     * @param puertoDestino Puerto de destino
     * @param fechaFin Fecha de fin
     * @param tipoTransporte Tipo de transporte del trayecto
     */
    protected TrayectoSimple(Muelle muelleOrigen, Puerto puertoOrigen, LocalDate fechaInicio, 
                          Muelle muelleDestino, Puerto puertoDestino, LocalDate fechaFin, String tipoTransporte) { // MODIFICADO
        super(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin, tipoTransporte); // MODIFICADO
    }


    @Override
    public String obtenerInformacionCompleta() {
        return "Trayecto simple desde " + getPuertoOrigen().getLocalidad() + " hasta " + getPuertoDestino().getLocalidad()  
        		+ " (" + this.getClass().getSimpleName() + ")";
    }


}
