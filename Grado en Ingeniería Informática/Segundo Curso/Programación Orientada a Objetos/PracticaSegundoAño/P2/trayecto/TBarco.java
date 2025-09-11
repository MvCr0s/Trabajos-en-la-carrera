package trayecto;

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;

import muelle.Muelle;
import puerto.Puerto;


/**
 * Clase TBarco representa un trayecto simple realizado en barco.
 * Hereda de la clase TrayectoSimple.
 * El coste del trayecto se calcula en función de la duración en días y un coste fijo por día.
 * @author alfdelv
 * @author mardedi
 */
public class TBarco extends TrayectoSimple {

    private static final double COSTE_POR_DIA = 4000;

    public TBarco(Muelle muelleOrigen, Puerto puertoOrigen, LocalDate fechaInicio, 
                  Muelle muelleDestino, Puerto puertoDestino, LocalDate fechaFin) {
        super(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin, "barco");
    }

    @Override
    public double calcularPrecioTrayecto() {
        long diasTrayecto = ChronoUnit.DAYS.between(getFechaInicio(), getFechaFin());
        return diasTrayecto * COSTE_POR_DIA;
    }

}