package ordenadores;

import java.util.ArrayList;

public class impresoraLaser extends componentes {
	private String tipoToner;
	private int paginas_impresas;
	
	public impresoraLaser(String fabricante,String modelo,double pVenta,String tipoConector,ArrayList<Integer> puertosValidos,String tipoToner, int paginas_impresas) {
		super(fabricante,modelo,pVenta,tipoConector,puertosValidos,'S');
		this.setTipoToner(tipoToner);
		this.setPaginas_impresas(paginas_impresas);
	}

	public int getPaginas_impresas() {
		return paginas_impresas;
	}

	public void setPaginas_impresas(int paginas_impresas) {
		this.paginas_impresas = paginas_impresas;
	}

	public String getTipoToner() {
		return tipoToner;
	}

	public void setTipoToner(String tipoToner) {
		this.tipoToner = tipoToner;
	}

}
