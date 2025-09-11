package trayecto;



/**
 * Representa un trayecto combinado entre dos trayectos simples.
 * Este es un caso específico donde siempre hay exactamente dos trayectos: 
 * uno terrestre (camión) y otro adicional (barco o tren).
 * @author alfdelv
 * @author mardedi
 */
public abstract class TrayectoCombinado extends Trayecto {
    private TrayectoSimple trayecto1;
    private TrayectoSimple trayecto2;

    /**
     * Constructor de un trayecto combinado que une dos trayectos simples.
     * 
     * @param trayecto1 Primer trayecto simple
     * @param trayecto2 Segundo trayecto simple
     * @throws IllegalArgumentException Si los trayectos no son consecutivos o tienen distinto tipo de transporte
     */
    protected TrayectoCombinado(TrayectoSimple trayecto1, TrayectoSimple trayecto2) {
        super(
            trayecto1.getMuelleOrigen(),   // Muelle de origen del primer trayecto
            trayecto1.getPuertoOrigen(),  // Puerto de origen del primer trayecto
            trayecto1.getFechaInicio(),   // Fecha de inicio del primer trayecto
            trayecto2.getMuelleDestino(), // Muelle de destino del segundo trayecto
            trayecto2.getPuertoDestino(), // Puerto de destino del segundo trayecto
            trayecto2.getFechaFin(),      // Fecha de fin del segundo trayecto
            "Combinado"                   // Tipo de transporte del combinado (representativo)
        );

        if (!trayecto1.getMuelleDestino().equals(trayecto2.getMuelleOrigen())) {
            throw new IllegalArgumentException("El destino del primer trayecto debe coincidir con el origen del segundo trayecto.");
        }

        this.trayecto1 = trayecto1;
        this.trayecto2 = trayecto2;
    }


    /**
     * Obtiene el primer trayecto del combinado.
     * 
     * @return Primer trayecto simple.
     */
    public TrayectoSimple getTrayecto1() {
        return trayecto1;
    }

    /**
     * Obtiene el segundo trayecto del combinado.
     * 
     * @return Segundo trayecto simple.
     */
    public TrayectoSimple getTrayecto2() {
        return trayecto2;
    }

    /**
     * Método abstracto que calcula el precio del trayecto combinado.
     * Cada tipo de trayecto combinado (como el pack Camión-Barco o Camión-Tren)
     * tendrá su propia lógica de descuento o ajuste en el precio.
     *
     * @return El precio total del trayecto combinado
     */
    public abstract double calcularPrecioTrayecto();

    
    /**
     * Obtiene un resumen simple de la información de los trayectos combinados.
     *
     * @return Un resumen del tipo de pack y los trayectos.
     */
    @Override
    public String obtenerInformacionCompleta() {
        return "Trayecto combinado entre " + getPuertoOrigen().getLocalidad() + " y " 
            + getPuertoDestino().getLocalidad() + " (" + this.getClass().getSimpleName() + ")";
    }
}
