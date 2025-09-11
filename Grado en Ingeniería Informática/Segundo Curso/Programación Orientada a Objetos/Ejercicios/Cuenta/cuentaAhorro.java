package Cuenta;

public class cuentaAhorro extends cuenta {
	private float ingresoPact;
	private float saldo;
	
	public cuentaAhorro(Cliente titular) {
		super(titular);
	}
	
	public void reintegro(float cantidad) {
		assert(saldo>0);
		if(saldo>1000) {
			saldo-=cantidad+ingresoPact;
		}else{
			saldo-=cantidad;
		}
	}
}
