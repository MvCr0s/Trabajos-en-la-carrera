package uva.tds;

import java.util.ArrayList;

/**
 * Clase que representa una agenda que gestiona contactos.
 * Cada contacto está compuesto por un nombre y un apellido.
 * 
 * @author mardedi
 */
public class Agenda {

	private ArrayList<Contacto> contactos;

	/**
	 * Constructor de la clase Agenda.
	 * Inicializa una lista vacía de contactos.
	 */
	public Agenda() {
		this.contactos = new ArrayList<>();
	}

	/**
	 * Añade un contacto a la agenda.
	 * 
	 * @param contacto El contacto a añadir.
	 * @throws IllegalArgumentException si el contacto es nulo.
	 */
	public void addContacto(Contacto contacto) {
		if (contacto == null) {
			throw new IllegalArgumentException("El contacto no puede ser nulo.");
		}
		contactos.add(contacto);
	}

	/**
	 * Obtiene un contacto de la agenda que coincida con el nombre proporcionado.
	 * 
	 * @param nombre El nombre del contacto a buscar.
	 * @return El contacto encontrado o {@code null} si no existe.
	 */
	public Contacto getContacto(String nombre) {
		for (Contacto contacto : contactos) {
			if (contacto.getNombre().equals(nombre)) {
				return contacto;
			}
		}
		return new ContactoNulo();
	}

	/**
	 * Modifica el apellido de un contacto con el nombre proporcionado.
	 * 
	 * @param nombre        El nombre del contacto a modificar.
	 * @param nuevoApellido El nuevo apellido a asignar.
	 * @throws IllegalArgumentException si el nombre no existe o si el nuevo apellido es nulo o vacío.
	 */
	public void modificarApellido(String nombre, String nuevoApellido) {
		if (nuevoApellido == null || nuevoApellido.isEmpty()) {
			throw new IllegalArgumentException("El nuevo apellido no puede ser nulo o vacío.");
		}

		for (Contacto contacto : contactos) {
			if (contacto.getNombre().equals(nombre)) {
				contacto.setApellido(nuevoApellido);
				return;
			}
		}
		throw new IllegalArgumentException("No existe un contacto con el nombre proporcionado: " + nombre);

	}

}
