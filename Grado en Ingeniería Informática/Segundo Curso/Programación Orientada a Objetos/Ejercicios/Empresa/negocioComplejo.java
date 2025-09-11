package Empresa;

import java.util.ArrayList;

public class negocioComplejo extends negocio {
	private ArrayList<negocio> negocios;

	public negocioComplejo(persona gerente, long benef, long inversion, int numContratos) {
		super(gerente, benef, inversion, numContratos);
		this.negocios=negocios;
		
	}
	public void añadirNegocio(negocio i){
		assert(i!=null);
		negocios.add(i);
	}
	public float getBeneficios() {
		float benef=0;
		for(negocio i : negocios) {
			benef+=i.getBenef();
		}
		return benef;
	}
	
	public int getNumContratos() {
		int contratos=0;
		for(negocio i : negocios) {
			contratos+=i.getNumContratos();
		}
		return contratos/negocios.size();
	}
	
	public long getInversion() {
		long inversion=0;
		for(negocio i : negocios) {
			inversion+=i.getInversion();
		}
		return inversion;
	}
}
