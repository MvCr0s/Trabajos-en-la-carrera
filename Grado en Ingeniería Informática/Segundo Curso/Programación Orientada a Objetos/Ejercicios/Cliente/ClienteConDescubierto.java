package Cliente;

public class ClienteConDescubierto extends Clientes{
	
	public ClienteConDescubierto(Persona cliente) {
		super(cliente);
	}
	
	public float[] getPagos() {
		float[] pagos=new float[3];
		pagos[0]=(float) (0.5*getSaldoDispuesto());
		pagos[1]=(float) (0.25*getSaldoDispuesto());
		pagos[2]=(float) (0.25*getSaldoDispuesto());
		return pagos;
	}

}
