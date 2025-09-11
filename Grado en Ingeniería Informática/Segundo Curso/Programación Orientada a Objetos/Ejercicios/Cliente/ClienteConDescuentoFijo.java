package Cliente;

public class ClienteConDescuentoFijo extends Clientes {
	private Persona persona;
	private float descuento;

	public ClienteConDescuentoFijo(Persona persona,float descuento) {
		super(persona);
		this.descuento=descuento;
	}

	@Override
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
	

}
