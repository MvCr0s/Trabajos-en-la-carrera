package Empresa;

public class empresa {
	private persona presidente;
	private String CIF;
	private String direccionPostal;
	
	public empresa(persona presidente,String CIF,String direccionPostal) {
		this.presidente=presidente;
		this.CIF=CIF;
		this.direccionPostal=direccionPostal;
	}

	public persona getPresidente() {
		return presidente;
	}

	public void setPresidente(persona presidente) {
		this.presidente = presidente;
	}

	public String getCIF() {
		return CIF;
	}

	public void setCIF(String cIF) {
		CIF = cIF;
	}

	public String getDireccionPostal() {
		return direccionPostal;
	}

	public void setDireccionPostal(String direccionPostal) {
		this.direccionPostal = direccionPostal;
	}

}
