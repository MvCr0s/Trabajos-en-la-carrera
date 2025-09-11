package empleado;

import java.time.LocalDate;

public class empleadoFijo extends Empleado {
	private LocalDate a�oDeAlta;
	private double complementoFijo;
	private double precio;
	
	public empleadoFijo(String nombre, int edad, String nif, LocalDate a�oDeAlta, double complementoFijo,double precio) {
		super(nombre,edad,nif);
		this.a�oDeAlta=a�oDeAlta;
		this.complementoFijo=complementoFijo;
		this.precio=precio;
	}

	public LocalDate getA�oDeAlta() {
		return a�oDeAlta;
	}
	
	public double getcomplementoFijo() {
		return complementoFijo;
	}

	@Override
	public double calcularSueldo() {
		return getcomplementoFijo()+precio;		
	}

}
