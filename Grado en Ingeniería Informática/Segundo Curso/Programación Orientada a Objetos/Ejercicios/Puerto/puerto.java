package Puerto;

import java.time.LocalDate;
import static java.time.temporal.ChronoUnit.DAYS;

public class puerto {
	private String nombre;
	private String DNI;
	private LocalDate inicio;
	private LocalDate fin;
	private String posicionAmarre;
	public double eslora;
	public LocalDate a�o;
	
	public puerto(String nombre,String DNI,LocalDate inicio, LocalDate fin,String posicionAmarre,double eslora,LocalDate a�o) {
		this.setNombre(nombre);
		this.setDNI(DNI);
		this.setInicio(inicio);
		this.setFin(fin);
		this.setPosicionAmarre(posicionAmarre);
		this.eslora=eslora;
		this.a�o=a�o;
	}
	
	public double getPrecio() {
		double dias=DAYS.between(inicio,fin);
		return dias*eslora*10*12;
	}

	public String getPosicionAmarre() {
		return posicionAmarre;
	}

	public void setPosicionAmarre(String posicionAmarre) {
		this.posicionAmarre = posicionAmarre;
	}

	public LocalDate getFin() {
		return fin;
	}

	public void setFin(LocalDate fin) {
		this.fin = fin;
	}

	public LocalDate getInicio() {
		return inicio;
	}

	public void setInicio(LocalDate inicio) {
		this.inicio = inicio;
	}

	public String getDNI() {
		return DNI;
	}

	public void setDNI(String dNI) {
		DNI = dNI;
	}

	public String getNombre() {
		return nombre;
	}

	public void setNombre(String nombre) {
		this.nombre = nombre;
	}
	

}
