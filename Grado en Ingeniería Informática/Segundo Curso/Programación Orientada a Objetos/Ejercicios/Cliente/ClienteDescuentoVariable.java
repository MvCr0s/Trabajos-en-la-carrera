package Cliente;

public class ClienteDescuentoVariable extends Clientes{
	private Persona cliente;
	private float descuento;
	
	public ClienteDescuentoVariable(Persona cliente,float descuento) {
		super(cliente);
		this.setDescuento(descuento);
	}
	
	public float[] getPagos() {
		float pagos[]=new float[3];
		pagos[0]=1/3*getSaldoDispuesto();
		pagos[1]=1/3*getSaldoDispuesto();
		pagos[2]=1/3*getSaldoDispuesto();
		return pagos;
	}

	public float getDescuento() {
		return descuento;
	}

	public void setDescuento(float descuento) {
		this.descuento = descuento;
	}

}
