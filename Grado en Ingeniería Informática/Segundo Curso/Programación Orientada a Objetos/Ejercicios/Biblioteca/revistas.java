package Biblioteca;

import java.time.LocalDate;

public class revistas extends documentos {
	private double volumen;
	private int numero;
	private LocalDate salida;

	public revistas(String autor, String titulo, String editorial, LocalDate fechaPublicacion, String estado,double volumen,int numero,LocalDate salida) {
		super(autor, titulo, editorial, fechaPublicacion, estado);
		this.numero=numero;
		this.volumen=volumen;
		this.salida=salida;
		
	}
	
	@Override
	public boolean puedePrestar(persona p) {
		if(!p.getNombre().equals("Alumno")) {
			return true;
		}
		return false;
	}

	public LocalDate getSalida() {
		return salida;
	}

	public int getNumero() {
		return numero;
	}

	public double getVolumen() {
		return volumen;
	}

}
