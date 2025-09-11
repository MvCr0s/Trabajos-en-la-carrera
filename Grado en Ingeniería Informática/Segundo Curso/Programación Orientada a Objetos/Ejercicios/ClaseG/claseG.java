package ClaseG;

import java.util.ArrayList;

public class claseG<G> {
	private ArrayList<G> ge = new ArrayList<>();
	
	public claseG() {
		this.ge=new ArrayList<>();
	}
	
	
	 public void agrega(G elemento) {
		 assert(elemento!=null);
		 ge.add(elemento);
	 }
	 public void elimina(G elemento) {
		 assert(elemento!=null);
		 assert(ge.contains(elemento));
		 ge.remove(elemento);
	 }
	 public boolean esta(G elemento) {
		 assert(elemento!=null);
		 if(ge.contains(elemento)) {
			 return true;
		 }
		 return false;
	 }
	 public boolean esVacio() {
		 if(ge.isEmpty()) {
			 return false;
		 }
		 return true;
	 }
	 public int getCardinal() {
		 return ge.size();
	 }

	 
	  public claseG<G> union(claseG<G> otroConjunto) {
	        claseG<G> resultado = new claseG<>();
	        resultado.ge.addAll(ge);
	        resultado.ge.addAll(otroConjunto.ge);
	        return resultado;
	    }
	  
	 public claseG<G> interseccion(claseG<G> otroConjunto){
		 claseG<G> resultado = new claseG<>();
		 for (G i : ge) {
			 if(otroConjunto.esta(i)) {
				 resultado.agrega(i);
			 }
		 }
		 return resultado;
	 }
	 
	    public claseG<G> diferencia(claseG<G> otroConjunto) {
	    	claseG<G> resultado = new claseG<>();
	        for (G elemento : this.ge) {
	            if (!otroConjunto.esta(elemento)) {
	                resultado.agrega(elemento);
	            }
	        }
	        return resultado;
	    }

}
