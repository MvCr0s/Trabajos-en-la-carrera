package empleado;

import java.time.LocalDate;

public class empleadoHoras extends Empleado{
	private int numHoras;
	private int precio;

	public empleadoHoras(String nombre, int edad, String nif, int numHoras,int precio) {
		super(nombre, edad, nif);
		this.numHoras=numHoras;
		this.precio=precio;
		
	}

	public int getNumHoras() {
		return numHoras;
	}
	
	public void setNumHoras(int numHoras) {
		this.numHoras=numHoras;
	}

	public int getPrecio() {
		return precio;
	}

	@Override
	public double calcularSueldo() {
		return numHoras*precio;
	}
}
