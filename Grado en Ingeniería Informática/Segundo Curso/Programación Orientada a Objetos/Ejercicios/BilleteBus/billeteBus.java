package BilleteBus;

import java.time.LocalDate;

import java.time.LocalTime;

public class billeteBus{
	private String salida;
	private String llegada;
	private LocalTime fechayhora;
	
	public billeteBus(String salida,String llegada, LocalTime fechayhora) {
		assert(salida!=null);
		assert(llegada!=null);
		assert(fechayhora!=null);
		this.llegada=llegada;
		this.salida=salida;
		this.fechayhora=fechayhora;
	}
	
	public String getLlegada() {
		return llegada;
	}
	public String getSalida() {
		return salida;
	}
	
	public double getPrecio() {
		return 20;
	}
	
	public LocalTime getFechayhora() {
		return fechayhora;
	}
	
	@Override
	public String toString() {
		return "El billete de"+getSalida()+"a"+getLlegada()+"tiene un precio de"+getPrecio()+"para"+getFechayhora() ;
	}

}
