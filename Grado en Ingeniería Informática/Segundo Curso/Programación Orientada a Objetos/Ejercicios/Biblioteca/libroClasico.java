package Biblioteca;

import java.time.LocalDate;

public class libroClasico extends documentos{
	private String tipo;

	public libroClasico(String autor,String titulo,String editorial,LocalDate fechaPublicacion,String estado,String tipo) {
		super(autor,titulo,editorial,fechaPublicacion,estado);
		this.tipo=tipo;
	}

	@Override
	public boolean puedePrestar(persona p) {
		if(p.getNombre().equals("Alumno") || getTipo().equals("Diccionario") || getTipo().equals("ISO")) {
			return false;
		}
		return true;
	}

	public String getTipo() {
		return tipo;
	}

}
