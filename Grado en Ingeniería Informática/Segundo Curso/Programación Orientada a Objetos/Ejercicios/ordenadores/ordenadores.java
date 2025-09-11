package ordenadores;

import java.util.ArrayList;

public class ordenadores {
	private componentes unidadCentral;
	private ArrayList<componentes> EyS; 
	
	public ordenadores(componentes unidadCentral, ArrayList<componentes> EyS) {
		this.unidadCentral=unidadCentral;
		this.EyS=EyS;
	}
	
	public void setUnidadCentral(componentes unidadCentral) {
		this.unidadCentral=unidadCentral;		
	}
	
	public componentes getUnidadCentral(componentes unidadCentral) {
		return unidadCentral;
	}
	
	public void setEyS(ArrayList<componentes> EyS){
		this.EyS=EyS;
	}
	
	public void getEyS(ArrayList<componentes> EyS){
		this.EyS=EyS;
	}
	
	public void añadirEyS(componentes perif) {
		if(EyS.contains(perif)) {
			throw new IllegalArgumentException();
		}else {
			EyS.add(perif);
		}
	}
	
	public void eliminarEyS(componentes perif) {
		if(!EyS.contains(perif)) {
			throw new IllegalArgumentException();
		}else if(EyS.size()>2) {
			EyS.remove(perif);
		}else {
			throw new IllegalArgumentException();
		}
	}
	
	public boolean componente(ArrayList<componentes> EyS) {
		boolean e=false;
		boolean s=false;
		
		if(EyS.size()<2) {
			return false;
		}
		for(componentes i : EyS) {
			if (i.getTipo()=='E') {
				e=true;
			}else if(i.getTipo()=='S') {
				s=true;
			}
		}
		return e&s;
	}
	
	public double Pventa(componentes unidadCentral, ArrayList<componentes> EyS) {
		double pVenta=0;
		pVenta=unidadCentral.getpVenta();
		for(componentes i : EyS) {
			pVenta+=i.getpVenta();
		}	
		return pVenta;
	}
}
