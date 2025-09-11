package Cliente;

public abstract class Clientes implements Cliente {
	private Persona persona;
	private float saldoDispuesto;
	private float cargoAutorizado;
	
	public Clientes(Persona persona) {
        this.persona = persona;
    }

	public Persona getPersonaCliente() {
		return persona;
	}
	public float getSaldoDispuesto() {
		return saldoDispuesto;
	}
	public float getCargoAutorizado() {
		return cargoAutorizado;
	}
	public void cargar(float importe) {
		saldoDispuesto+=getCargoAutorizado();
	}
	public abstract float[] getPagos();

}
