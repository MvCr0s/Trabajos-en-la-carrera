package empleado;

import java.time.LocalDate;

public class empleadoTemporal extends Empleado {
	private LocalDate fechaA;
	private LocalDate fechaB;

	public empleadoTemporal(String nombre, int edad, String nif, LocalDate fechaA, LocalDate fechaB) {
		super(nombre, edad, nif);
		this.setFechaA(fechaA);
		this.setFechaB(fechaB);
		
	}

	public LocalDate getFechaA() {
		return fechaA;
	}

	public void setFechaA(LocalDate fechaA) {
		this.fechaA = fechaA;
	}

	public LocalDate getFechaB() {
		return fechaB;
	}

	public void setFechaB(LocalDate fechaB) {
		this.fechaB = fechaB;
	}

	@Override
	public double calcularSueldo() {
		double sueldo = 1000;
		return sueldo;
	}

}
