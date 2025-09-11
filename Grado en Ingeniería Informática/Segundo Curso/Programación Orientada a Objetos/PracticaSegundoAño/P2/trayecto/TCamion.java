package trayecto;

import java.time.LocalDate;

import muelle.Muelle;
import puerto.Puerto;

/**
 * Clase TCamion representa un trayecto simple realizado en camión.
 * Hereda de la clase TrayectoSimple.
 * El coste del trayecto se calcula en función de la distancia recorrida (en kilómetros) y un coste fijo.
 * @author alfdelv
 * @author mardedi
 */
public class TCamion extends TrayectoSimple {

    private static final double COSTE_POR_KM = 4.5; 
    private static final double COSTE_FIJO = 200;    

    public TCamion(Muelle muelleOrigen, Puerto puertoOrigen, LocalDate fechaInicio, 
                   Muelle muelleDestino, Puerto puertoDestino, LocalDate fechaFin) {
        super(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin, "camion");
    }

    @Override
    public double calcularPrecioTrayecto() {
        double distancia = obtenerDistanciaMillasMarinas() * 1.60934;  
        return COSTE_FIJO + (distancia * COSTE_POR_KM);
    }

}
