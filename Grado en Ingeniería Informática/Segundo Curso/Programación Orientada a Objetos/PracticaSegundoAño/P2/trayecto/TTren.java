package trayecto;

import java.time.LocalDate;

import muelle.Muelle;
import puerto.Puerto;


/**
 * Clase TTren representa un trayecto simple realizado en tren.
 * Hereda de la clase TrayectoSimple.
 * El coste del trayecto se calcula en función de la distancia recorrida (en kilómetros) y un coste fijo adicional.
 * @author alfdelv
 * @author mardedi
 */
public class TTren extends TrayectoSimple {

    private static final double COSTE_FIJO = 20;  
    private static final double COSTE_POR_KM = 12.5;  

    public TTren(Muelle muelleOrigen, Puerto puertoOrigen, LocalDate fechaInicio, 
                 Muelle muelleDestino, Puerto puertoDestino, LocalDate fechaFin) {
        super(muelleOrigen, puertoOrigen, fechaInicio, muelleDestino, puertoDestino, fechaFin, "tren");
    }

    @Override
    public double calcularPrecioTrayecto() {
        double distancia = obtenerDistanciaMillasMarinas() * 1.60934; 
        return COSTE_FIJO + (distancia * COSTE_POR_KM);
    }

}
