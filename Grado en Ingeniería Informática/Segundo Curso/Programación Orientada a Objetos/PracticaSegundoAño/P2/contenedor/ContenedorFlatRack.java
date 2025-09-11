package contenedor;

/**
 * Clase ContenedorFlatRack representa un contenedor tipo Flat Rack.
 * Este tipo de contenedor no es apilable y requiere dos plazas debido a su diseño.
 * Hereda de la clase Contenedor y redefine comportamientos específicos.
 * @author alfdelv
 * @author mardedi
 */
public class ContenedorFlatRack extends Contenedor {
	
	public int plazasRequeridas = 2;
	
    public ContenedorFlatRack(String id, double peso, double cargaMaxima, double capacidad, Estado estado) {
        super(id, peso, cargaMaxima, capacidad, false, estado);
    }

    @Override
    public boolean puedeApilar() {
        return false;
    }

    @Override
    public boolean esCompatibleConInfraestructura(String infraestructura) {

        return infraestructura.equalsIgnoreCase("barco") ||
               infraestructura.equalsIgnoreCase("tren");
    }
    
    @Override
    public void setTecho(boolean techo) {
        if (techo) {
            throw new UnsupportedOperationException("Un contenedor FlatRack no puede tener techo.");
        }
        super.setTecho(techo); 
    }
}