package Cliente;

public interface Cliente {
	public Cliente(Persona persona);
	public Persona getPersonaCliente();
	public float getSaldoDispuesto();
	public float getCargoAutorizado();
	public void cargar(float importe);

}
