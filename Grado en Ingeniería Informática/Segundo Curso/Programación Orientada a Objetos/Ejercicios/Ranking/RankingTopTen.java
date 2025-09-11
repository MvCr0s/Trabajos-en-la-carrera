package Ranking;

import java.util.ArrayList;

public class  RankingTopTen<E>{
	private ArrayList<E> ranking = new ArrayList<>();
	private ArrayList<E> nominados = new ArrayList<>();
	private int[] votos = new int[10];
	private boolean votacionesCerradas;

	public RankingTopTen(){
		ranking = new ArrayList<>();
		nominados = new ArrayList<>();
		votos = new int[10];	
		this.votacionesCerradas=false;
	}

/**
* Añade un elemento al ArrayList nominados indicando que ese elemento ha sido nominado
*@param el elemento que queremos nominar
*@AssertionError si el elemento es nulo
*@AssertionError si ya hay 10 elementos nominados
*@AssertionError si ese elemento ya ha sido nominado
*@AssertionError si ya se han cerrado las votaciones
*/

	public void sePuedeNominar(E element){
		assert(element != null);
		assert(nominados.size()<11);
		assert(!nominados.contains(element));
		assert(!votacionesCerradas);
		if(nominados.isEmpty()){
			nominados.add(element);
		}else {
			assert(nominados.get(0).getClass().equals(element));
			nominados.add(element);
		}
	}

/**
*Iniciamos el array de votos a 0 porque todavia no se han empezado las votaciones
*/
	public void iniciarVotos(){
		for(int i =0; i<10;i++){
			votos[i]=0;
		}
	}

/**
*Sumamos 1 al array de votos en la posicion cuyo indice equivale a ese mismo elemento en el ArrayList de nominados
*@param el elemento al que se quiere votar
*@AssertionError si el elemento es nulo
*@AssertionError si ese elemento no ha sido nominado
*@AssertionError si ya se han cerrado las votaciones
*/

	public void Votar(E element){
		assert(element != null);
		assert(nominados.contains(element));
		assert(!votacionesCerradas);
		iniciarVotos();
		for(int i=0;i<nominados.size();i++){
			if(nominados.get(i).equals(element)){
				votos[i]=votos[i]+1;
			}
		}
	}

/**
*Permite saber el numero de elementos que han sido nominados
*@return el numero de elementos nominados en ese momento
*/
	
	public int elementosNominados(){
		return nominados.size();
	}

/**
*permite saber si un determinado elemento ha sido nominado
*@return True si ese elemento esta nominado, false en caso contrario
*/

	public boolean elementoEstaNominados(E element){
		assert(element!=null);
		return nominados.contains(element);
	}

/**
*Permite finalizar el periodo de votaciones y nominaciones
*/

	public void cerrarVotaciones(){
		this.votacionesCerradas=true;
	}

/**
*Permite saber el numero de votos de un elemento en un determinado momento
*@AssertionError si el elemento es nulo
*@AssertionError si las votaciones no se han cerrado
*@AssertionError si ese elemento no ha sido nominado
*@param el elemento del que queremos saber los votos
*@return el numero total de votos de ese elemento
*/
	public int numVotos(E element){
		assert(element!=null);
		assert(votacionesCerradas);
		assert(nominados.contains(element));
		int voto=0;
		for(int i=0;i<nominados.size();i++){
			if(nominados.get(i).equals(element)){
				voto=votos[i];
			}
		}
		return voto;

	}

/**
*Permite saber el ranking de los elementos nominados
*@AssertionError si las votaciones todavia no se han cerrado
*@return el ranking de los nominados 
*/
	
	public ArrayList<E> ranking(){
		assert(votacionesCerradas);
		while(ranking.size()!=nominados.size()){
			int numVotos=0;
			int index=0;
			for(int i=0;i<votos.length;i++){
				if(votos[i]>numVotos){
					numVotos=votos[i];
					index=i;
				}
			}
			votos[index]=-1;
			ranking.add(nominados.get(index));			 
		}
		return ranking;
	}

}