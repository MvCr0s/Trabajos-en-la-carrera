package ordenadores;

import java.util.ArrayList;

public class impresora extends componentes{
	
	private String tipoToner;
	
	public impresora(String fabricante,String modelo,double pVenta,String tipoConector,ArrayList<Integer> puertosValidos,String tipoToner) {
		super(fabricante,modelo,pVenta,tipoConector,puertosValidos,'S');
		this.setTipoToner(tipoToner);
	}

	public String getTipoToner() {
		return tipoToner;
	}

	public void setTipoToner(String tipoToner) {
		this.tipoToner = tipoToner;
	}
	
}
