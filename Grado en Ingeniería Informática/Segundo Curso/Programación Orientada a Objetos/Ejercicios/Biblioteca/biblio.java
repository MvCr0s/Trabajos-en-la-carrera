package Biblioteca;

import java.time.LocalDate;

public class biblio {
	private int numDocumento;
	private LocalDate fecha;
	private String DNI;
	
	public biblio(int numDocumento,LocalDate fecha,String DNI) {
		this.numDocumento=numDocumento;
		this.fecha=fecha;
		this.DNI=DNI;
	}

	public int getNumDocumento() {
		return numDocumento;
	}

	public void setNumDocumento(int numDocumento) {
		this.numDocumento = numDocumento;
	}

	public LocalDate getFecha() {
		return fecha;
	}

	public void setFecha(LocalDate fecha) {
		this.fecha = fecha;
	}

	public String getDNI() {
		return DNI;
	}

	public void setDNI(String dNI) {
		DNI = dNI;
	}

}
