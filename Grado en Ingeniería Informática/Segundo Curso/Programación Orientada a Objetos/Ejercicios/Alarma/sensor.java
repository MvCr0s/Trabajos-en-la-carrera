package Alarma;

public class sensor {
	private boolean sensor;
	private double medida;
	private double umbral;
	
	public sensor(boolean sensor,double medida){
		this.setMedida(medida);
		this.sensor=false;		
	}

	public boolean isSensor() {
		return sensor;
	}

	public void setSensor(boolean sensor) {
		this.sensor = sensor;
	}

	public double getMedida() {
		return medida;
	}

	public void setMedida(double medida) {
		this.medida = medida;
	}

	public double getUmbral() {
		return umbral;
	}

	public void setUmbral(double umbral) {
		this.umbral = umbral;
	}
	
	public boolean ALARMA() {
		if(umbral<medida) {
			return true;
		}
		else {
			return false;
		}
	}
}
