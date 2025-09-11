package contenedor;

/**
 * Clase ContenedorEstandar representa un contenedor estándar.
 * Hereda de la clase Contenedor y define su comportamiento específico en relación con la apilabilidad
 * y la compatibilidad con diferentes tipos de infraestructura.
 * @author alfdelv
 * @author mardedi
 */
public class ContenedorEstandar extends Contenedor {

	public ContenedorEstandar(String nCodigo, double nPesoTara, double nCargaMaxima, double nVolumen, boolean nTecho, Estado nEstado) {

        super(nCodigo, nPesoTara, nCargaMaxima, nVolumen, nTecho, nEstado);
    }


    @Override
    public boolean puedeApilar() {
        return true; 
    }

    @Override
    public boolean esCompatibleConInfraestructura(String infraestructura) {

        return infraestructura.equalsIgnoreCase("barco") ||
               infraestructura.equalsIgnoreCase("tren") ||
               infraestructura.equalsIgnoreCase("camion");
    }
}