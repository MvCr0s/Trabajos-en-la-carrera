package BilleteBus;

import java.time.LocalTime;

public class BilleteInfantil extends billeteBus{
	private Persona p;
	private String nombreYapellido;
	public BilleteInfantil(String salida,String llegada,LocalTime fechayhora,Persona p,String nombreYapellido){
		super(salida,llegada,fechayhora);
		assert(p.getEdad()<12);
		this.p=p;
		this.nombreYapellido=nombreYapellido;
	}
	
	
	public boolean mismaPers(String nombre) {
		if(getNombreYapellido().equals(nombre)){
			return true;
		}
		return false;
	}
	
	public double getPrecio() {
		return getPrecio()*0.5;
	}
		
	@Override
	public String toString() {
		return "El billete de"+getSalida()+"a"+getLlegada()+"tiene un precio de"+getPrecio()+"para"+getNombreYapellido()+"de edad"+p.getEdad();
	}


	public String getNombreYapellido() {
		return nombreYapellido;
	}
}
