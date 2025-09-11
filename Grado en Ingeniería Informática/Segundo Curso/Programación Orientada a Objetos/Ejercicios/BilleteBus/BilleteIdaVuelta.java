package BilleteBus;

import java.util.ArrayList;


public class BilleteIdaVuelta{
	billeteBus billete1;
	billeteBus billete2;
	public BilleteIdaVuelta(billeteBus ida, billeteBus vuelta) {		
		//COMPROBACION BILLETES

		billete1=ida;
		billete2=vuelta;
		
	}
	@Override
	public String toString() {
		return "sale y llega: " + billete1.getLlegada() ;
	}
}

/**
 * OTRA IMPLEMENTACION
 * 
 * public class GoldenAgePack{

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
 */

