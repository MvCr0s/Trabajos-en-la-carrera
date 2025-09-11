

import java.util.HashMap;

public class Door implements Thing {
  /* *********************
   * Nombres de parametros
   * ********************* */
  public final String PlaceId = Constantes.PlaceId; // destino puerta
  public final String Name    = Constantes.DoorName;
  public final String Message = Constantes.DoorMessage;
  public final String MP      = Constantes.DoorMP;
  public final String HP      = Constantes.DoorHP;
  public final String Event   = Constantes.DoorEvent;
  public final String Status  = Constantes.DoorStatus;

  private String placeId;
  private String name;
  private String message;
  private int    mp;
  private int    hp;
  private String event;
  private String status;

  public Door(final String placeId) {
    this.placeId = placeId;
    name         = Constantes.DefaultDoorName;
    message      = Constantes.DefaultDoorMessage;
    hp           = Constantes.DefaultDoorHP;
    mp           = Constantes.DefaultDoorMP;
    event        = Constantes.DefaultDoorEvent;
    status       = Constantes.DefaultDoorStatus;
  }
  public Door(final String placeId, String name, String message,
                 int hp, int mp, String event) {
    this.placeId = placeId;
    this.name    = name;
    this.message = message;
    this.hp      = hp;
    this.mp      = mp;
    this.event   = event;
  }

  @Override
  public java.util.Map<String, String> toMap() {
    java.util.Map<String, String> valor =
      java.util.Map.of(
         PlaceId, new String(placeId),
         Name,    new String(name),
         Message, new String(message),
         MP,      Integer.toString(mp),
         HP,      Integer.toString(hp),
         Status,  new String(status),
         Event,   event);
    return valor;
  }

  /* **
   * Setters especiales
   */
  @Override
  public void setNameMessage(String name, String message) {
    this.name    = new String(name);
    this.message = new String(message);
  }

  public void setHpMp(int hp, int mp) {
    this.hp     = hp;
    this.mp     = mp;
  }

  @Override
  public void openIt() {      // Abre la puerta
    this.status = Constantes.StatusOpened;
  }

  @Override
  public void closeIt() {     // cierra si se recierra
    if (event.equals(Constantes.EventRecloses)) {
      this.status = Constantes.StatusClosed;
    }
  }

  @Override
  public void setEvent(String ev) {
    switch(ev) {
    case Constantes.EventReopens:
      this.event = Constantes.EventReopens; break;
    case Constantes.EventRecloses:
      this.event = Constantes.EventRecloses;
    }
  }

  /* **
   * Getters
   */
  @Override
  public String getName()    { return name; }
  public String getDestino() { return placeId; }
  @Override
  public String getMessage() { return message; }
  @Override
  public int    getHP()      { return hp; }
  @Override
  public int    getMP()      { return mp; }
  @Override
  public String getEvent()   { return event; }
  @Override
  public String getStatus()  { return status; }
}
