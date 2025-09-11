package contenedor;

import java.util.ArrayList;

import trayecto.Trayecto;

/**
 * Representa un contenedor que almacena información como peso, volumen y trayectos.
 * @author alfdelv
 * @author mardedi
 *
 */
public abstract class Contenedor {
	
	/**
	 * Define el estado del contenedor: en tránsito o recogido.
	 * @author fredi
	 *
	 */
	public enum Estado { 	
	    TRANSITO, RECOGIDA
	}
	
	private String codigo;
	private double pesoTara;
	private double volumen;
	protected boolean techo;
	protected Estado estado;
	private double cargaMaxima; 
	ArrayList<Trayecto> trayectos = new ArrayList<>();
	

	/**
	 * Crea un nuevo contenedor con los atributos especificados.
	 * @param nCodigo El código del contenedor
     * @param nPeso_tara El peso en tara del contenedor
     * @param ncargaMaxima La carga máxima del contenedor
     * @param nVolumen El volumen del contenedor
     * @param nTecho Indica si el contenedor tiene techo
     * @param nEstado El estado inicial del contenedor
	 * @throws IllegalArgumentException cuando el código del contenedor no es váildo
	 */
	protected Contenedor(String nCodigo, double nPesoTara, double ncargaMaxima, double nVolumen, boolean nTecho, Estado nEstado) {
		setPesoTara(nPesoTara);
		setcargaMaxima(ncargaMaxima);
		setVolumen(nVolumen);
		setCodigo(nCodigo);
		setTecho(nTecho);
		setEstado(nEstado);
		
	}

	/**
	 * Obtiene el código del contenedor.
     * @return El código del contenedor
	 */
	public String getCodigo() {
		return this.codigo;
	}

	/**
     * Establece el código del contenedor.
     * @param codigo El nuevo código del contenedor
     * @throws IllegalArgumentException cuando el código del contenedor no es váildo
     */
	public void setCodigo(String codigo) {
		if(comprobarCodigo(codigo)) {
			this.codigo = codigo;
		}else {
			throw new IllegalArgumentException("El código no cumple con el estandar ISO 6346");
		}
	}

	/**
     * Obtiene el peso en tara del contenedor.
     * @return El peso en tara
     */
	public double getPesoTara() {
		return pesoTara;
	}

	/**
     * Establece el peso en tara del contenedor.
     * @param pesoTara El nuevo peso en tara
     */
	public void setPesoTara(double pesoTara) {
		this.pesoTara = pesoTara;
	}

	/**
     * Obtiene la carga máxima del contenedor.
     * @return La carga máxima
     */
	public double getcargaMaxima() {
		return cargaMaxima;
	}

	/**
     * Establece la carga máxima del contenedor.
     * @param ncargaMaxima La nueva carga máxima
     */
	public void setcargaMaxima(double ncargaMaxima) {
		this.cargaMaxima=ncargaMaxima;
	}

    /**
     * Obtiene el volumen del contenedor.
     * @return El volumen del contenedor
     */
	public double getVolumen() {
		return volumen;
	}

	/**
     * Establece el volumen del contenedor.
     * @param volumen El nuevo volumen
     */
	public void setVolumen(double volumen) {
		this.volumen = volumen;
	}
	
	/**
     * Indica si el contenedor tiene techo.
     * @return True si tiene techo, false si no
     */
	public boolean getTecho() {
		return techo;
	}
	
	/**
     * Establece si el contenedor tiene techo o no.
     * @param techo El estado del techo
     */
	public void setTecho(boolean techo) {
		this.techo = techo;
	}
	
	/**
     * Obtiene el estado actual del contenedor.
     * @return El estado actual
     */
	public Estado getEstado() {
		return estado;
	}

	/**
     * Establece el estado del contenedor.
     * @param estado El nuevo estado
     */
	public void setEstado(Estado estado) {
		this.estado = estado;
	}
	
	/**
     * Cambia el estado del contenedor a "RECOGIDA".
     */
	public void setEstadoARecogida() {
		setEstado(Estado.RECOGIDA);
	}
	/**
     * Cambia el estado del contenedor a "TRANSITO".
     */
	public void setEstadoATransito() {
		setEstado(Estado.TRANSITO);
	}
	
	/**
     * Obtiene el volumen en metros cúbicos.
     * @return El volumen en metros cúbicos
     */
	public double getVolumenM3() {
		return(getVolumen());
	}
	
	/**
     * Obtiene el volumen en pies cúbicos.
     * @return El volumen en pies cúbicos
     */
	public double getVolumenFt3() {
		return(getVolumen()*35.3147);
	}
	
	/**
     * Obtiene el peso en kilogramos.
     * @return El peso en kilogramos
     */
	public double getPesoKg() {
		return(getPesoTara());
	}
	
	/**
     * Obtiene el peso en libras. 
     * @return El peso en libras
     */
	public double getPesoLb() {
		return(getPesoTara()*2.20462);
	}
	
	/**
     * Añade un trayecto a la lista de trayectos del contenedor.
     * @param nTrayecto El trayecto a añadir
     */
	public void addTrayecto(Trayecto nTrayecto) {
		trayectos.add(nTrayecto);
	}
	
	/**
     * Calcula el precio total de todos los trayectos del contenedor.
     * @return El precio total de los trayectos
     */
	public double getPrecioTrayectos() {
		double total = 0.0;
        for (Trayecto trayecto : trayectos) {
			total += trayecto.calcularPrecioTrayecto();
        }
        return total;
	}
	
	private boolean comprobarCodigo(String nCodigo) {
	    double numA = 0;

	    if ((nCodigo.length() == 11) && (nCodigo.matches("^[A-Z]{3}.*")) &&
	        (nCodigo.charAt(3) == 'U' || nCodigo.charAt(3) == 'J' || nCodigo.charAt(3) == 'Z')) {

	        for (int i = 0; i < 4; i++) {
	            numA += conversion(nCodigo.charAt(i)) * Math.pow(2, i);
	        }
	        for (int i = 4; i < 10; i++) {
	            numA += Character.getNumericValue(nCodigo.charAt(i)) * Math.pow(2, i);
	        }

	        double numB = numA / 11;
	        int numC = (int) numB;
	        int numD = numC * 11;
	        int control = (int) numA - numD;

	        return control == Character.getNumericValue(nCodigo.charAt(10));
	    }
	    
	    return false;
	}
	
	private int conversion(char letra) {
	    int[] valores = {10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38};

	    if (letra >= 'A' && letra <= 'Z') {
	        return valores[letra - 'A' ];
	    }

	    return 0;
	}
	
	
	/*************************************************************************
	 * 								PRACTICA 2								 *
	 *************************************************************************/
	/**
	 * Metodo que comprueba si un contenedor se puede apilar.
	 * @return True si se puede apilar, false para lo contrario.
	 */
	public abstract boolean puedeApilar();
	
	public abstract boolean esCompatibleConInfraestructura(String infraestructura);
	 

}
