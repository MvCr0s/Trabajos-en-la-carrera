package Puerto;

import java.time.LocalDate;

public class velero extends barco{
	private int mastiles;

	public velero(String m, double tamano, LocalDate a, int mastiles) {
		super(m, tamano, a);
		this.setMastiles(mastiles);
	}
	
	public double getModulo() {
		return getEslora()*10+getMastiles();
	}

	public int getMastiles() {
		return mastiles;
	}

	public void setMastiles(int mastiles) {
		this.mastiles = mastiles;
	}
	
}
