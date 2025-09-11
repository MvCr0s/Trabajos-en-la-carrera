package empleado;

public abstract class Empleado {
	 private String nombre;
	 private int edad;
	 private String nif;
	
	public Empleado(String nombre, int edad, String nif) {
		this.nombre=nombre;
		this.edad=edad;
		this.nif=nif;
		
	}
	
	public abstract double calcularSueldo();
	
	public int getEdad() {
		return edad;
	}

	public String getNif() {
		return nif;
	}

	public String getNombre() {
		return nombre;
	}

	@Override
	public String toString() {
		return ""+getNombre()+","+getEdad()+","+getNif();
	}

}
