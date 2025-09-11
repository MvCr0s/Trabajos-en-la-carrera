

import java.util.HashMap;
import java.util.ArrayList;

/**
 * Mapa - Singleton
 *
 * ya veremos que se hace con esto mas adelante.
 */

/*
 * Esto quedaria mucho mas bonito si se leyera de un archivo
 * Json o un CSV, sencillamente. Pero queria verificar que
 * funcionaran las clases.
 */

public class Mapa {
  public HashMap<String, Place> lugares;
  public Place start;

  public Mapa() {
    this.lugares = new HashMap<>();
  }

  // getter que devuelve el punto de inicio del mapa.
  public Place start() {
    return start;
  }

  // donde va una puerta.
  public Place to(String nombreLugar){
    return lugares.get(nombreLugar);
  }

  public void init() {
    // this.lugares = new HashMap<>();

    /* Aquí ponemos el array con los nombres de las
     * habitaciones ~~(que aquí son sólo números).~~
     */
    String [] vectorId = {
      "hall", "pasillo", "dormitorio", "sala", "patio", "cocina", "sotano"
    };

    for (String id : vectorId) {
      lugares.put(id, new Place(id));
    }

    /* Mejor no ser tan criptico */
    Place            lugar;

    /* Aquí arrancamos */
    start = lugares.get("hall");
    /* Lugar "hall" */
      lugar = start;
      lugar.setNameMessage("HALL", "Aqui empieza todo");
      lugar.addDoor(new Door("pasillo",    "a", "Puerta a.", 20, 20, Constantes.EventReopens));
      lugar.addDoor(new Door("dormitorio", "b", "Puerta b.", 200, 200, Constantes.EventReopens));
      lugar.addChest(new Chest("A", "Cofre A.", 0, 0, -30, -30, Constantes.EventReopens));

    /* Lugar "pasillo" */
      lugar = lugares.get("pasillo");
      lugar.setNameMessage("Pasillo", "Largo pasillo");
      lugar.addDoor(new Door("hall",  "a", "Puerta a.", 30, 40, Constantes.EventReopens));
      lugar.addDoor(new Door("sala",  "b", "Puerta b.", 5, 5, Constantes.EventRecloses));
      lugar.addDoor(new Door("patio", "c", "Puerta c.", 10, 15, Constantes.EventRecloses));
      // no hay cofres

    /* Lugar "dormitorio" - habitacion del tesoro */
      lugar = lugares.get("dormitorio");
      lugar.setNameMessage("Dormitorio", "Gran salon sin salida");
      // no hay salida
      lugar.addChest(new Chest("A", "Gran Cofre A.", 20, 30, 100, 100, Constantes.EventEndsGame));

    /* Lugar "sala": salita de pasar */
      lugar = lugares.get("sala");
      lugar.setNameMessage("Salon", "Salon de estar");
      lugar.addDoor(new Door("pasillo",    "a", "Puerta a.", 10, 5, Constantes.EventReopens));
      lugar.addChest(new Chest("A", "Cofre A.", 5, 0, 2, 20, Constantes.EventRecloses));
      lugar.addChest(new Chest("B", "Cofre B.", 0, 5, 15, 15, Constantes.EventRecloses));
    /* Lugar "patio" */
      lugar = lugares.get("patio");
      lugar.setNameMessage("Patio", "Gran Patio con columnas");
      lugar.addDoor(new Door("pasillo",    "a", "Puerta a.", 20, 20, Constantes.EventReopens));
      lugar.addDoor(new Door("cocina",    "b", "Puerta b.", 10, 5, Constantes.EventRecloses));
      lugar.addChest(new Chest("A", "Cofre A.", 10, 0, 30, 30, Constantes.EventRecloses));
      lugar.addChest(new Chest("B", "Cofre B.", 5, 5, 5, 20, Constantes.EventReopens));
      lugar.addChest(new Chest("C", "Cofre C.", 0, 10, 20, 30, Constantes.EventRecloses));
    /* Lugar "cocina" */
      lugar = lugares.get("cocina");
      lugar.setNameMessage("Cocina", "Buen lugar caliente");
      lugar.addDoor(new Door("sala", "a", "Puerta a.", 20, 0, Constantes.EventReopens));
      lugar.addDoor(new Door("sotano", "b", "Puerta b.", 10, 0, Constantes.EventRecloses));
      lugar.addChest(new Chest("A", "Cofre A.", 20, 0, 30, 10, Constantes.EventRecloses));
      lugar.addChest(new Chest("B", "Cofre B.", 10, 10, 15, 20, Constantes.EventRecloses));


    /* Lugar "sotano" */
      lugar = lugares.get("sotano");
      lugar.setNameMessage("Sotano", "Lugar obscuro de paso");
      lugar.addDoor(new Door("pasillo", "a", "Puerta a.", 5, 0, Constantes.EventReopens));
      lugar.addChest(new Chest("A", "Cofre A.", 2, 2, 15, 15, Constantes.EventRecloses));

      // return true;
  }
}
