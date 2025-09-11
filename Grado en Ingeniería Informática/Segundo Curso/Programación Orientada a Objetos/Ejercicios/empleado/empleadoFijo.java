package empleado;

import java.time.LocalDate;

public class empleadoFijo extends Empleado {
	private LocalDate añoDeAlta;
	private double complementoFijo;
	private double precio;
	
	public empleadoFijo(String nombre, int edad, String nif, LocalDate añoDeAlta, double complementoFijo,double precio) {
		super(nombre,edad,nif);
		this.añoDeAlta=añoDeAlta;
		this.complementoFijo=complementoFijo;
		this.precio=precio;
	}

	public LocalDate getAñoDeAlta() {
		return añoDeAlta;
	}
	
	public double getcomplementoFijo() {
		return complementoFijo;
	}

	@Override
	public double calcularSueldo() {
		return getcomplementoFijo()+precio;		
	}

}
