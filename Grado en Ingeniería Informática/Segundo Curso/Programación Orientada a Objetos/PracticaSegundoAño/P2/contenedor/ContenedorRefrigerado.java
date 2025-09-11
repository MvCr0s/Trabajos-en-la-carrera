package contenedor;

/**
 * Clase ContenedorRefrigerado representa un contenedor equipado con un sistema de refrigeración.
 * Este tipo de contenedor requiere estar conectado a una fuente de energía para mantener la refrigeración.
 * Hereda de la clase Contenedor y añade funcionalidades específicas relacionadas con la refrigeración.
 * @author alfdelv
 * @author mardedi
 */
public class ContenedorRefrigerado extends Contenedor {
    private boolean conectado;


    public ContenedorRefrigerado(String id, double peso, double capacidad, double cargaMaxima, Estado estado) {
        super(id, peso, cargaMaxima, capacidad, true, estado);
        this.conectado = false;
    }

    /**
     * Conecta el motor del contenedor a una fuente de energía.
     */
    public void conectar() {
        this.conectado = true;
    }

    /**
     * Desconecta el motor del contenedor de la fuente de energía.
     */
    public void desconectar() {
        this.conectado = false;
    }

    /**
     * Verifica si el contenedor está conectado a una fuente de energía.
     * @return True si está conectado, false si no
     */
    public boolean estaConectado() {
        return conectado;
    }

    @Override
    public boolean puedeApilar() {
        return true;
    }
    
    @Override
    public void setTecho(boolean techo) {
        if (!techo) {
            throw new UnsupportedOperationException("Un contenedor refrigerado debe tener techo.");
        }
        super.setTecho(techo); 
    }

    @Override
    public boolean esCompatibleConInfraestructura(String infraestructura) {

        return infraestructura.equalsIgnoreCase("barco") ||
               infraestructura.equalsIgnoreCase("tren") ||
               infraestructura.equalsIgnoreCase("camion");
    }
}