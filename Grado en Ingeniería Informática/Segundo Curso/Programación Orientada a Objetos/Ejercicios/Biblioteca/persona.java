package Biblioteca;

public class persona {
	private String nombre;
	private String apellido;
	private String dni;
	
	public persona(String nombre, String apellido, String dni) {
		this.nombre=nombre;
		this.apellido=apellido;
		this.dni=dni;
	}
	public String getNombre() {
		return nombre;
	}
	public String getApellido() {
		return apellido;
	}
	public String getDNI() {
		return dni;
	}
	
}


