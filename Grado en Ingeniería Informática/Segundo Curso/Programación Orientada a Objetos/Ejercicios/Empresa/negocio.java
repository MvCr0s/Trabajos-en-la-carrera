package Empresa;

public class negocio {
	private persona gerente;
	private long benef;
	private long inversion;
	private int numContratos;
	
	public negocio(persona gerente,long benef,long inversion,int numContratos) {
		this.gerente=gerente;
		this.benef=benef;
		this.inversion=inversion;
		this.numContratos=numContratos;
	}

	public persona getGerente() {
		return gerente;
	}

	public void setGerente(persona gerente) {
		this.gerente = gerente;
	}

	public long getBenef() {
		return benef;
	}

	public void setBenef(long benef) {
		this.benef = benef;
	}

	public long getInversion() {
		return inversion;
	}

	public void setInversion(long inversion) {
		this.inversion = inversion;
	}

	public int getNumContratos() {
		return numContratos;
	}

	public void setNumContratos(int numContratos) {
		this.numContratos = numContratos;
	}
	
}
