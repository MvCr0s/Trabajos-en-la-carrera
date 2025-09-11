import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.HashMap;
import java.util.Map;

public class Juego extends UnicastRemoteObject implements JuegoInterface {
  private final Mapa mapa;
  private final Map<VistaCB, PersonajeImpl> jugadores;

  public Juego() throws RemoteException {
    super(); // Llamar al constructor de la superclase
    mapa = new Mapa();
    jugadores = new HashMap<>();
    mapa.init(); // Inicializar el mapa aquí
    System.out.println("Juego inicializado.");
  }

  @Override
  public Personaje newPlay(String jugador, String rol, VistaCB cb)
          throws RemoteException, BadPersonajeException {
    System.out.println("Nuevo juego iniciado por: " + jugador + " con el rol: " + rol);
    PersonajeImpl personaje = new PersonajeImpl(jugador, rol);
    personaje.init(this);
    jugadores.put(cb, personaje);
    return personaje;
  }

  @Override
  public boolean quitPlay(VistaCB cb) throws RemoteException {
    System.out.println("Jugador ha salido del juego.");
    return jugadores.remove(cb) != null;
  }

  // Métodos para gestionar el mapa
  public Place start() {
    return mapa.start();
  }

  public Place to(String lugar) {
    return mapa.to(lugar);
  }
}
