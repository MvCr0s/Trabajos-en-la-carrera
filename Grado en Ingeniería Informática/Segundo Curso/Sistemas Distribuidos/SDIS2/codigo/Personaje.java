import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;
import java.util.Map;

public interface Personaje extends Remote {
  Map<String, String> info() throws RemoteException;
  Place here() throws RemoteException;
  String status() throws RemoteException;
  List<Accion> acciones() throws RemoteException;
  String actua(Accion accion) throws RemoteException, BadActionException;
}
