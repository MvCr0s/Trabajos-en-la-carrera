import java.rmi.Remote;
import java.rmi.RemoteException;

public interface VistaCB extends Remote {
    String updateView(String message) throws RemoteException;
}
