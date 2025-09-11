package Cuenta;

public class Cliente {
	private String nombre;
	private String apellido;
	private String dni;
	private int edad;
	
	
	public Cliente(String nombre, String apellido, String dni, int edad) {
		this.nombre=nombre;
		this.apellido=apellido;
		this.dni=dni;
		this.edad=edad;
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
	public int getEdad() {
		return edad;
	}
	
}
