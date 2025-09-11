package Cuenta;

public class cuenta {
	private Cliente titular;
	private float saldo;
	
	// parte privada de la clase
	public cuenta(Cliente titular) {
		this.titular=titular;
	}
	public Cliente getTitular() { // Cliente tiene un método getEdad
		return titular;
	} 

	public float getSaldo() {
		return saldo;
	}
	  
	// otros métodos
	/**
	** @assertion.pre (cantidad >=0 && getSaldo()-cantidad>=0)
	**/
	public void reintegro(float cantidad) {
		saldo-=cantidad;
	}

	/**
	** @assertion.pre (cantidad >=0)
	**/
	public void ingreso(float cantidad) {
		 saldo+=cantidad;
	 }
}