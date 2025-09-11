package trayecto;



/**
 * Clase PackCamionBarco representa un trayecto combinado que incluye un trayecto en camión
 * y otro en barco, aplicando un descuento del 15% en el precio del trayecto en barco.
 * Extiende la clase TrayectoCombinado.
 * @author alfdelv
 * @author mardedi
 */
public class PackCamionBarco extends TrayectoCombinado {

    public PackCamionBarco(TrayectoSimple camion, TrayectoSimple barco) {
        super(camion, barco);
        if (!(camion instanceof TCamion) || !(barco instanceof TBarco)) {
            throw new IllegalArgumentException("El pack debe incluir un trayecto en camión y otro en barco.");
        }
    }

    @Override
    public double calcularPrecioTrayecto() {
        double precioCamion = getTrayecto1().calcularPrecioTrayecto();
        double precioBarco = getTrayecto2().calcularPrecioTrayecto() * 0.85; 
        return precioCamion + precioBarco;
    }

    @Override
    public String obtenerInformacionCompleta() {
    	 String info = super.obtenerInformacionCompleta();
         return info + "Tipo de pack: Camión + Barco (con 15% de descuento en barco)\n";
    }
}
