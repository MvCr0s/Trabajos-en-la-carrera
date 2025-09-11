

import java.util.HashMap;

public class Place {
  /* *********************
   * Nombres de parametros
   * ********************* */
  public final String Id      = Constantes.PlaceId;
  public final String Name    = Constantes.PlaceName;
  public final String Message = Constantes.PlaceMessage;

  private HashMap<String, Door>  puertas ;
  private HashMap<String, Chest> cofres  ;
  private HashMap<String, Thing> cosas ;

  private final String id;
  private String name;
  private String message;

  public Place(final String id) {
    this.id      = id;
    this.name    = Constantes.DefaultPlaceName;
    this.message = Constantes.DefaultPlaceMessage;
    this.puertas = new HashMap<String, Door>();
    this.cofres  = new HashMap<String, Chest>();
    this.cosas   = new HashMap<String, Thing>();
  }

  public java.util.Map<String, String> toMap() {
    java.util.Map<String, String> valor =
      java.util.Map.of(
         Id,      new String(id),
         Name,    new String(name),
         Message, new String(message));
    return valor;
  }

  /* **
   * Setters especiales
   */
  public void setNameMessage(String name, String message) {
    this.name    = new String(name);
    this.message = new String(message);
  }
  public void addDoor(Door puerta) {
    this.puertas.put(puerta.getName(), puerta);
    this.cosas.put(puerta.getName(), (Thing) puerta);
  }
  public void addChest(Chest cofre) {
    this.cofres.put(cofre.getName(), cofre);
    this.cosas.put(cofre.getName(), (Thing) cofre);
  }

  /* * Renueva el lugar para la siguiente visita
   * Hay que usarla con cuidado porque hay que controlar que
   * no queda nadie. Es decir hay que usarla justo cuando
   * se va el ultimo jugador de la sala.
   */
  public void reset() {
    // Nota, se puede hacer mas simple sobre Things[]
    for (Chest cofre  : cofres.values())
      cofre.closeIt();
    for (Door  puerta : puertas.values())
      puerta.closeIt();
  }

  /* **
   * Getters
   */
  public String getName()    { return name; }
  public String getMessage() { return message; }
  
  public Door getDoor(String nombre) {
    return puertas.get(nombre) ;
  }
  public Chest getChest(String nombre) {
    return cofres.get(nombre) ;
  }
  public Thing getThing(String nombre) {
    return cosas.get(nombre) ;
  }
  /*
   * Lista de nombres de puertas y cofres de una estancia.
   */
  public HashMap<String,String> toList() {
    HashMap<String, String> lista = new HashMap<>();
    for (String nombre : cosas.keySet()) {
      lista.put(nombre, cosas.get(nombre).getMessage());
    }
    return lista;
  }
}
