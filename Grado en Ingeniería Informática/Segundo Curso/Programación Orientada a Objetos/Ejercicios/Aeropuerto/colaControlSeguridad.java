package Aeropuerto;

import java.util.ArrayList;

/**
 * Clase que representa una cola de control de seguridad en un aeropuerto.
 * Esta cola está diseñada para manejar elementos que extienden la clase PaAvion.
 *
 * @param <E> Tipo de elementos que pueden ser gestionados por la cola, debe extender PaAvion.
 */
public class colaControlSeguridad<E extends PaAvion> {
    /**
     * Lista que representa la cola de control de seguridad.
     */
    private ArrayList<E> cola = new ArrayList<>();
    private long tamMax;

    /**
     * Constructor que inicializa la cola de control de seguridad con una lista existente.
     *
     * @param cola Lista preexistente que se utilizará como cola de control de seguridad.
     */
    public colaControlSeguridad(long tamMax) {
    	cola = new ArrayList<>();
    	this.tamMax=tamMax;
    }

    /**
     * Introduce un elemento en la cola de control de seguridad.
     * El elemento debe ser válido según las reglas definidas por el método elementValido.
     *
     * @param element Elemento a introducir en la cola de control de seguridad.
     * @throws IllegalArgumentException Si el elemento no es válido.
     */
    public void introducirCola(E element) {
        if (!elementValido(element) || cola.size()>tamMax) {
            throw new IllegalArgumentException("Elemento no válido");
        }
        cola.add(element);
    }

    /**
     * Verifica si un elemento es válido para ser introducido en la cola de control de seguridad.
     * Un elemento es válido si es diferente de nulo y comparte la misma clase que el primer elemento
     * de la cola (si existe).
     *
     * @param element Elemento a verificar.
     * @return true si el elemento es válido, false de lo contrario.
     */
    public boolean elementValido(E element) {
        if (element == null) {
            throw new IllegalArgumentException("Elemento no puede ser nulo");
        }
        if (cola.isEmpty()) {
            return true;
        } else return element.getClass().equals(cola.get(0).getClass());
    }

    /**
     * Devuelve el siguiente elemento a ser atendido en la cola de control de seguridad.
     *
     * @return Elemento siguiente a ser atendido.
     */
    public E siguienteAtendido() {
        return cola.get(0);
    }

    /**
     * Elimina el primer elemento de la cola de control de seguridad.
     */
    public void eliminar() {
        cola.remove(cola.get(0));
    }

    /**
     * Cancela un vuelo eliminando los elementos asociados a ese vuelo de la cola de control de seguridad.
     *
     * @param numVuelo Número de vuelo a cancelar.
     */
    public void vueloCancelado(String numVuelo) {
        for (E element : cola) {
            if (element.getNumVuelo().equals(numVuelo)) {
                cola.remove(element);
                break;
            }
        }
    }

    /**
     * Adelanta un vuelo moviendo los elementos asociados a ese vuelo al principio de la cola de control de seguridad.
     *
     * @param numVuelo Número de vuelo a adelantar.
     */
    public void vueloAdelantado(String numVuelo) {
        ArrayList<E> aux = new ArrayList<>();
        for (E element : cola) {
            if (!element.getNumVuelo().equals(numVuelo)) {
                aux.add(element);
                cola.remove(element);
            }
        }
        cola.addAll(0, aux);
    }

    /**
     * Retrasa un vuelo moviendo los elementos asociados a ese vuelo al final de la cola de control de seguridad.
     *
     * @param numVuelo Número de vuelo a retrasar.
     */
    public void vueloRetrasado(String numVuelo) {
        ArrayList<E> aux = new ArrayList<>();
        for (E element : cola) {
            if (element.getNumVuelo().equals(numVuelo)) {
                aux.add(element);
                cola.remove(element);
            }
        }
        cola.addAll(aux);
    }
}
