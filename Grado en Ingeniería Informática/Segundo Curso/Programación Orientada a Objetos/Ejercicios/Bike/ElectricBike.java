package Bike;

public class ElectricBike extends Bike{
	private double v;
	private double ah;
	private double wh; 
	public ElectricBike(String id, String size,double v,double ah,double wh) {
		super(id,size);
		this.v=v;
		this.ah=ah;
		this.wh=wh;
	}

	public double getDepositToPay(double deposit){
		return deposit*(1+(getV()/100));
	}

	public void setSize(String size){
		assert(size!=null);
		this.size=size;
	}

	public void setV(double v){
		this.v=v;
	}

	public void setAh(double ah){
		this.ah=ah;
	}

	public void setWh(double wh){
		this.wh=wh;
	}

	public double getV(){
		return v;
	}

	public double getAh(){
		return ah;
	}

	public double getWh(){
		return getAh()+getV();
	}
	
	
	
	@Override
	public String toString() {
		return "id: " + getId() + ", " + "size: " + getSize()+ " , Voltaje:" + getV() + ", Carga electrica: "+ getAh() +" , Energia: " + getWh() ;
	}







}