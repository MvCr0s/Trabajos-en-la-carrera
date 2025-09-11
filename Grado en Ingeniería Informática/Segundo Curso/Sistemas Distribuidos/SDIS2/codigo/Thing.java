

/* **
 * Cosa: comun entre Door y Chest.
 */
public interface Thing {
  public java.util.Map<String,String> toMap();
  public void setNameMessage(String name, String message);
  public void setEvent(String ev);
  public void   openIt() ;
  public void   closeIt() ;

  public String getName() ;
  public String getMessage() ;
  public int    getHP() ;
  public int    getMP() ;
  public String getEvent() ;
  public String getStatus() ;
}


