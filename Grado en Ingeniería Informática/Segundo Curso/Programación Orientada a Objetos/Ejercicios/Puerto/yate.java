package Puerto;

import java.time.LocalDate;

public class yate extends barco {
	
	private double CV;
	
	public yate(String m, double tamano, LocalDate a, double CV) {
		super(m, tamano, a);
		this.setCV(CV);
		
	}
	
	public double getModulo() {
		return getEslora()*10+getCV();
	}

	public double getCV() {
		return CV;
	}

	public void setCV(double cV) {
		CV = cV;
	}



}
