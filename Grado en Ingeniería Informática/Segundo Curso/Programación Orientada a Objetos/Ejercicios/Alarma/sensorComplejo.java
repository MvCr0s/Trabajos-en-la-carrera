package Alarma;

import java.util.ArrayList;

import Alarma.sensor;

public class sensorComplejo{
	private ArrayList <sensor> sensores;
	private double umbral;
	
	public sensorComplejo(ArrayList <sensor> sensores,double umbral) {
		this.sensores=sensores;
		this.umbral=umbral;
	}
	
	public ArrayList <sensor> getSensores() {
		return sensores;
	}
	public void setSensores(ArrayList <sensor> sensores) {
		this.sensores = sensores;
	}
	
	public void a�adirSensores(sensor sensor) {
		if(sensores.contains(sensor)) {
			throw new IllegalArgumentException();
		}
		sensores.add(sensor);
	}
	
	public void eliminarSensores(sensor sensor) {
		if(sensores.contains(sensor)) {
			sensores.remove(sensor);
		}
		throw new IllegalArgumentException();
	}
	
	public int getNumSensores() {
		return sensores.size();
	}
	
	public boolean ALARMA() {
		double media=0;
		for(sensor i:sensores) {
			media+=i.getMedida();
		}
		if(media>umbral) {
			return true;
		}
		return false;
	}
}
