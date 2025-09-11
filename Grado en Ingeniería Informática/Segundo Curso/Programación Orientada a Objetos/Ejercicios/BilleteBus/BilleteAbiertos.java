package BilleteBus;

import java.time.LocalTime;

public class BilleteAbiertos extends billeteBus{
	private Persona p;
	private String nombreYapellido;
	
	public BilleteAbiertos(String salida,String llegada,LocalTime fechayhora, Persona p,String nombreYapellido) {
		super(salida,llegada,null);
		this.p=p;
		this.nombreYapellido=nombreYapellido;
	}
	
	public double getprecio() {
		if(!nombreYapellido.equals(null)) {
			return getprecio()+getprecio()*0.1;
		}
		return getprecio();

	}	

	public String getNombreYapellido() {
		return nombreYapellido;
	}
	
	@Override
	public String toString() {
		return "El billete de"+getSalida()+"a"+getLlegada()+"tiene un precio de"+getPrecio()+"para"+getNombreYapellido();
	}

	public Persona getP() {
		return p;
	}
}
