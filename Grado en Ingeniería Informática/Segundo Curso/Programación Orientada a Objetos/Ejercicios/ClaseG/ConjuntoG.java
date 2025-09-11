package ClaseG;

import java.util.ArrayList;

public class ConjuntoG <G>{
	
	private ArrayList<G> ge = new ArrayList<>(); 

	public ConjuntoG() {
		this.ge=ge;
	}
	
	public void agrega(G elemento){
		assert(elemento!=null);
		ge.add(elemento);
	}
	public void elimina(G elemento){
		assert(elemento!=null);
		ge.remove(elemento);
	}
	public boolean esta(G elemento){
		assert(elemento!=null);
		return ge.contains(elemento);
	}
	public boolean esVacio(){
		return ge.isEmpty();
	}
	public int getCardinal() {
		return ge.size();
	}

	public ConjuntoG<G> Union(ConjuntoG<G> otroConjunto){
		ConjuntoG<G> resultado = new ConjuntoG<>(); 
		assert(otroConjunto!=null);
        resultado.ge.addAll(ge);
        resultado.ge.addAll(otroConjunto.ge);
		return resultado;
	}
	
	public ConjuntoG<G> Interseccion(ConjuntoG<G> otroConjunto){
		ConjuntoG<G> resultado = new ConjuntoG<>(); 
		assert(otroConjunto!=null);
		for(G i : ge){
			if(otroConjunto.esta(i)){
				resultado.agrega(i);	
			}
		}
		return resultado;
	}

	public ConjuntoG<G> Diferencia(ConjuntoG<G> otroConjunto){
		ConjuntoG<G> resultado = new ConjuntoG<>(); 
		assert(otroConjunto!=null);
		for(G i : ge){
			if(!otroConjunto.esta(i)){
				resultado.agrega(i);	
			}
		}
		return resultado;
	}

}