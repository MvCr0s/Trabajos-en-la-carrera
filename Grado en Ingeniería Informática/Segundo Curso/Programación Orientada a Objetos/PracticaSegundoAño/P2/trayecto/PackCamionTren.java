package trayecto;

/**
 * Clase PackCamionTren representa un trayecto combinado que incluye un trayecto en camión
 * y otro en tren. La tarifa aplicada es de 10 euros por kilómetro, calculada a partir
 * de la distancia total en kilómetros (conversión de millas marinas).
 * Extiende la clase TrayectoCombinado.
 * @author alfdelv
 * @author mardedi
 */
public class PackCamionTren extends TrayectoCombinado {

    public PackCamionTren(TrayectoSimple camion, TrayectoSimple tren) {
        super(camion, tren);
        if (!(camion instanceof TCamion) || !(tren instanceof TTren)) {
            throw new IllegalArgumentException("El pack debe incluir un trayecto en camión y otro en tren.");
        }
    }

    @Override
    public double calcularPrecioTrayecto() {
        double kilometrosCamion = getTrayecto1().obtenerDistanciaMillasMarinas() * 1.60934; 
        double kilometrosTren = getTrayecto2().obtenerDistanciaMillasMarinas() * 1.60934; 
        return (kilometrosCamion + kilometrosTren) * 10; 
    }
    
    @Override
    public String obtenerInformacionCompleta() {
    	 String info = super.obtenerInformacionCompleta();
         return info + "Tipo de pack: Camión + Tren (con tarifa de 10 euros por kilómetro para tren)\n";
    }
}
