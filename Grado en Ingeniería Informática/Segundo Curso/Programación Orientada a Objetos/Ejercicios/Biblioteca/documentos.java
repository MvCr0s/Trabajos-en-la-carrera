package Biblioteca;

import java.time.LocalDate;

public abstract class documentos{
	private String autor;
	private String titulo;
	private String editorial;
	private LocalDate fechaPublicacion;
	private String estado;

	public documentos( String autor,String titulo,String editorial,LocalDate fechaPublicacion,String estado) {
		this.autor=autor;
		this.titulo=titulo;
		this.editorial=editorial;
		this.fechaPublicacion=fechaPublicacion;
		this.estado="Disponible";
		
	}

	public LocalDate getFechaPublicacion() {
		return fechaPublicacion;
	}

	public void setFechaPublicacion(LocalDate fechaPublicacion) {
		this.fechaPublicacion = fechaPublicacion;
	}

	public String getEditorial() {
		return editorial;
	}

	public void setEditorial(String editorial) {
		this.editorial = editorial;
	}

	public String getTitulo() {
		return titulo;
	}

	public void setTitulo(String titulo) {
		this.titulo = titulo;
	}

	public String getAutor() {
		return autor;
	}

	public void setAutor(String autor) {
		this.autor = autor;
	}

	public String getEstado() {
		return estado;
	}

	public void setEstado(String estado) {
		if(!estado.equals("Disponible") || !estado.equals("Prestado") ||!estado.equals("Reservado") ) {
				throw new IllegalArgumentException("estado no v�lido");
		}
		this.estado=estado;
	}
	
	public abstract boolean puedePrestar(persona p);
}
