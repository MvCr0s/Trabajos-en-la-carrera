package Cuenta;

public class cuentaCredito extends cuenta{
	private float comision;
	private float fijo;
	private float saldo;
	
	public cuentaCredito(Cliente titular) {
		super(titular);
	}
	
	public void reintegro(float cantidad) {
		assert(saldo>fijo);
		saldo-=cantidad-comision;
	}
}
