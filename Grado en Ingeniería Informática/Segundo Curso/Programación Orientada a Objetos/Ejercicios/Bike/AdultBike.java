package Bike;

public class AdultBike extends Bike{


	public AdultBike(String id, String size){
		super(id,size);
	}

	public void setSize(String size){
		assert(size!=null);
		this.size=size;
	}

	public double getDepositToPay(double deposit){
		return deposit;
	}
	
	@Override
	public String toString() {
		return "id: " + getId() + ", " + "size: " + getSize();
	}

}