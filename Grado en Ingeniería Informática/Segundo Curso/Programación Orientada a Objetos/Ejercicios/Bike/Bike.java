package Bike;

public abstract class Bike implements Resource {
		private String id;
		protected String size;
	
	public Bike(String id, String size) {
		setId(id);
		setSize(size);
	}
	public String getId() {
		return id;
	}
	public String getSize() {
		return size;
	}
	@Override
	public String toString() {
		return "id: " + getId() + ", " + "size: " + getSize();
	}
	protected abstract void setSize(String size);
	private void setId(String id) { /** ... **/}
}