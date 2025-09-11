package Cuenta;

public class cuentaJoven extends cuenta{
	private float saldo;
	
	public cuentaJoven(Cliente titular) {
		super(titular);
		assert(titular.getEdad()<25);
		
	}
	
	public void reintegro(float cantidad) {
		saldo-=cantidad;
	}
}
