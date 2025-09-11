package Puerto;

import java.time.LocalDate;

public class motor extends yate{

	private int camarotes;
	
	public motor(String m, double tamano, LocalDate a, double CV,int camarotes) {
		super(m, tamano, a, CV);
		this.setCamarotes(camarotes);
		
	}
	
	public double getModulo() {
		return getEslora()*10+getCV()+getCamarotes();
	}

	public int getCamarotes() {
		return camarotes;
	}

	public void setCamarotes(int camarotes) {
		this.camarotes = camarotes;
	}

}
