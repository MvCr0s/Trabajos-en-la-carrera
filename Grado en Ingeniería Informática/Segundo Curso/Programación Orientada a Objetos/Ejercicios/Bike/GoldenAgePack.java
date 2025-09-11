package Bike;

import java.util.ArrayList;

public class GoldenAgePack{

	private ArrayList<Bike> bicis = new ArrayList<>();

	public GoldenAgePack(ArrayList<Bike> bicis){
		this.bicis=bicis;
	}

	public void añadirBici(Bike b){
		assert(b!=null);
		assert(bicis.size()>9);
		assert(!bicis.contains(b));
		bicis.add(b);
	}

	public void eliminarBici(Bike b){
		assert(b!=null);
		assert(bicis.size()>10);
		assert(bicis.contains(b));
		bicis.remove(b);
		
	}

	public int numBicis(){
		return bicis.size();
	}

	public double getDepositToPay(double deposit){
		for(Bike i : bicis){
			if(i !=null){
				deposit+=deposit;
			}
		}

		return deposit*0.6;
	}

		
	@Override
	public String toString() {
		return "numero de bicis: " + numBicis();
	}

}
