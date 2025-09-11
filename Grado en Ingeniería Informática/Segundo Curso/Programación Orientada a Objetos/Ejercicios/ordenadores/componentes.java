package ordenadores;

import java.util.ArrayList;

public class componentes {
	private String fabricante;
	private String modelo;
	private double pVenta;
	private String tipoConector;
	private ArrayList<Integer> puertosValidos;
	private char tipo;
	
	public componentes(String fabricante,String modelo,double pVenta,String tipoConector,ArrayList<Integer> puertosValidos,char tipo) {
		this.fabricante=fabricante;
		this.modelo=modelo;
		this.pVenta=pVenta;
		this.tipoConector=tipoConector;		
		this.puertosValidos=puertosValidos;
		this.tipo=tipo;
	}
	
	public String getFabricante() {
		return fabricante;
	}
	
	public void setFabricante(String fabricante) {
		this.fabricante=fabricante;
	}
	
	public String getModelo() {
		return modelo;
	}
	
	public void setModelo(String modelo) {
		this.modelo=modelo;
	}

	public double getpVenta() {
		return pVenta;
	}
	
	public void setPventa(double pVenta) {
		this.pVenta=pVenta;
	}
	
	public String getTipoConector() {
		return tipoConector;
	}
	
	public void setTipoConector(String tipoConector) {
		this.tipoConector=tipoConector;
	}
	
	public void setTipo(char tipo) {
		this.tipo=tipo;
	}
	
	public char getTipo() {
		return tipo;
	}
	
	public ArrayList getPuertosValidos() {
		return puertosValidos;
	}
	
	public void setPuertosValidos(Integer puertoValido) {
		if(puertoValido==null || puertosValidos.contains(puertoValido)) {
			throw new IllegalArgumentException();
		}else {
			puertosValidos.add(puertoValido);
		}
	}

	
}
