

import java.util.HashMap;

public class Chest implements Thing {
  /* *********************
   * Nombres de parametros
   * ********************* */
  public final String Name    = Constantes.ChestName;
  public final String Message = Constantes.ChestMessage;
  public final String MP      = Constantes.ChestMP;
  public final String HP      = Constantes.ChestHP;
  public final String Food    = Constantes.ChestFood;
  public final String Potion  = Constantes.ChestPotion;
  public final String Event   = Constantes.ChestEvent;
  public final String Status  = Constantes.ChestStatus;

  private String name;
  private String message;
  private int    mp;
  private int    hp;
  private int    food;
  private int    potion;
  private String event;
  private String status;
  private int    foodNow;
  private int    potionNow;

  private Chest() {
    name    = Constantes.DefaultChestName;
    message = Constantes.DefaultChestMessage;
    hp      = Constantes.DefaultChestHP;
    mp      = Constantes.DefaultChestMP;
    food    = Constantes.DefaultChestFood;
    potion  = Constantes.DefaultChestPotion;
    event   = Constantes.DefaultChestEvent;
    event   = Constantes.DefaultChestStatus;
  }

  public Chest(String name, String message, int hp, int mp,
            int food, int potion, String event) {
    this.name    = name;
    this.message = message;
    this.hp      = hp;
    this.mp      = mp;
    this.food    = food;
    this.potion  = potion;
    this.foodNow   = food;
    this.potionNow = potion;
    this.event   = event;
  }

  @Override
  public java.util.Map<String, String> toMap() {
    java.util.Map<String, String> valor =
      java.util.Map.of(
         Name,    new String(name),
         Message, new String(message),
         MP,      Integer.toString(mp),
         HP,      Integer.toString(hp),
         Food,    Integer.toString(foodNow),
         Potion,  Integer.toString(potionNow),
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

  public void setHpMpFoodPotion(int hp, int mp, int food, int potion) {
    this.hp     = hp;
    this.mp     = mp;
    this.food   = food;
    this.potion = potion;
    this.foodNow   = food;
    this.potionNow = potion;
  }

  @Override
  public void setEvent(String ev) {
    switch(ev) {
    case Constantes.EventReopens:
      this.event = Constantes.EventReopens; break;
    case Constantes.EventRecloses:
      this.event = Constantes.EventRecloses;
    case Constantes.EventEndsGame:
      this.event = Constantes.EventEndsGame;
    }
  }
  
  @Override
  public void openIt() {    // abre el cofre
    status      = Constantes.StatusOpened;
    this.foodNow   = 0;
    this.potionNow = 0;
  }

  @Override
  public void closeIt() {   // cierra si se recierra
    if (event.equals(Constantes.EventRecloses)) {
      this.foodNow   = this.food;
      this.potionNow = this.potion;
      this.status = Constantes.StatusClosed;
    }
  }

  /* **
   * Getters
   */
  @Override
  public String getName()      { return name; }
  @Override
  public String getMessage()   { return message; }
  @Override
  public int    getHP()        { return hp; }
  @Override
  public int    getMP()        { return mp; }
  public int    getFood()      { return food; }
  public int    getFoodNow()   { return foodNow; }
  public int    getPotion()    { return potion; }
  public int    getPotionNow() { return potionNow; }
  @Override
  public String getEvent()     { return event; }
  @Override
  public String getStatus()    { return status;}
}
