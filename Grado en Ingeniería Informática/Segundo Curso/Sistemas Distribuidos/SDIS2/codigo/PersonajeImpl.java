

import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.HashMap;


public class PersonajeImpl extends UnicastRemoteObject implements Personaje {
  private String nombre; // nombre del Jugador
  private PersonajeStats stats; // estatus del personaje ahora
  private Juego          juego;
  private Place          lugar; // donde esta el personaje ahora

  /* Comienza con un lugar null: se inicializa con init() */
  public PersonajeImpl(String nombre, String rol)throws BadPersonajeException, RemoteException  {
    switch (rol) {
      case "Fighter" :
        stats = PersonajeStats.createFighter();
        break;
      case "Wizard" :
        stats = PersonajeStats.createWizard();
        break;
      case "Thief" :
        stats = PersonajeStats.createThief();
        break;
      default :
        throw new BadPersonajeException("Fighter Wizard Thief");
    }
    this.nombre = new String(nombre);
    this.lugar  = null;
  }

  public void init(Juego juego) throws RemoteException{
    this.juego = juego;
      this.lugar = juego.start();
  }
    
  /** letrero general sobre el personaje */
  @Override
  public java.util.Map<String,String> info() throws RemoteException {
    return stats.toMap();
  }

  /** muestra donde esta el personaje */
  public Place here() {
    return lugar;
  }


  /** definicion del personaje: title, stats, limits, skills...*/
  public String status() throws RemoteException{
    return "HP: "+stats.getHP()
          +"MP: "+stats.getMP() ;
  }

  /** muestra las acciones de que dispone PersonajeStats
   * en este momento y sus valores
   * A estas hay que añadir, Exit(etc) y Help.
   */
  public java.util.ArrayList<Accion> acciones() throws RemoteException{
    java.util.ArrayList<Accion> acciones = new java.util.ArrayList<>();
    HashMap<String,String> lugarToList;

    acciones.add(new Accion("Query", "Me"));
    acciones.add(new Accion("Query", "Place"));
    lugarToList = lugar.toList();
    for (String nombre : lugarToList.keySet()) {
      acciones.add(new Accion("Query", nombre));
      acciones.add(new Accion("Open",  nombre));
    }
    return acciones;
  }

  /** aplica en presente jugada un valor a una opcion,  */
  public String actua(Accion accion) throws BadActionException,  RemoteException {
    String respuesta;

    switch (accion.op) {
      case "Query" :
        switch (accion.arg) {
          case "Me" :
            respuesta =  "Hello "+nombre+", you are a\n"
                        +stats.getTitleSmall()
                        +"and your current stats are HP: "
                        +stats.getHP()+" and MP: "
                        +stats.getMP();
            System.out.println(" query me: "+accion);
	    return respuesta;
          case "Place" :
            System.out.println(" query place: "+accion);
            respuesta = "<"+lugar.getName()+">\n"
                       +lugar.getMessage()+"\n";
            HashMap<String,String> lista = lugar.toList();
            for (String nombre : lista.keySet())
               respuesta += "("+nombre+", "+lista.get(nombre)+")\n";
	    return respuesta;
      default:
        System.out.println(" query algo: " + accion);
        Thing cosa = lugar.getThing(accion.arg);
        if (cosa == null) {
            return "No se encontró la cosa especificada: " + accion.arg;
        }
        respuesta = "<" + cosa.getName() + ">\n"
                + cosa.getMessage() + "\n"
                + "HP: " + cosa.getHP()
                + ", MP: " + cosa.getMP()
                + cosa.getStatus();
      return respuesta;
        }
      case "Open" :
        Door puerta = lugar.getDoor(accion.arg);
        System.out.println(" open algo: "+accion);
        if (null != puerta) {
          // comprobamos si ya esta abierta
          if (Constantes.StatusOpened.equals(puerta.getStatus())){
            // traspasamos el umbral y reseteamos.
              Place nuevoLugar = null;
              nuevoLugar = juego.to(puerta.getDestino());
              lugar.reset();
	    lugar = nuevoLugar;
	    respuesta = "\"As you wish!...\"\n";
	    return respuesta;
	  } else {
          // comprobamos si tiene parametros suficientes
	    int hpEficaz = (puerta.getHP()*100)/stats.getAA();
	    int mpEficaz = (puerta.getMP()*100)/stats.getAA();

	    if (stats.getHP() >= hpEficaz && stats.getMP() >= mpEficaz) {
              stats.setHP(-hpEficaz); // descontamos
              stats.setMP(-mpEficaz);
              // traspasamos el umbral y reseteamos.
            Place nuevoLugar = null;
            nuevoLugar = juego.to(puerta.getDestino());
            puerta.openIt();
              lugar.reset();  
	      lugar = nuevoLugar;
	      respuesta="\"As you wish!...\"\n";
	      return respuesta;
	    } else {
              respuesta="\"Sorry, you don't have enough MP or HP!!\"\n"; 
	      return respuesta;
	    }
	  }
	} else {
	  Chest cofre = lugar.getChest(accion.arg);
	  if (null == cofre) { // no es puerta ni cofre !!??
            throw new BadActionException();
	  } // else...
          // comprobamos si ya esta abierto
          if (Constantes.StatusOpened.equals(cofre.getStatus())){
            // mensaje de cofre ya explotado
	    respuesta = "\"Just opened, I guess...\"\n";
	    return respuesta;
	  } else {
          // comprobamos si tiene parametros suficientes
	    int hpEficaz = (cofre.getHP()*100)/stats.getAA();
	    int mpEficaz = (cofre.getMP()*100)/stats.getAA();

	    if (stats.getHP() >= hpEficaz && stats.getMP() >= mpEficaz) {
              stats.setHP(-hpEficaz); // descontamos
              stats.setMP(-mpEficaz);
              // reaprovechamos las variables

	      hpEficaz = (cofre.getFoodNow()*stats.getFacHP())/100;
	      mpEficaz = (cofre.getPotionNow()*stats.getFacMP())/100;

	      respuesta="\"As you wish!...\"\n";
	      if (hpEficaz < 0)
		respuesta += "Rotten food... ";
	      if (mpEficaz < 0)
		respuesta += "Cursed Potion...";
              respuesta += "Food: "+hpEficaz+", Potion: "+mpEficaz;

	      cofre.openIt();
	      return respuesta;
	    } else {
              respuesta="\"Sorry, you don't have enough MP or HP!!\"\n"; 
	      return respuesta;
	    }
	  }
	}
      default :
	throw new BadActionException();
    }
  }
}
