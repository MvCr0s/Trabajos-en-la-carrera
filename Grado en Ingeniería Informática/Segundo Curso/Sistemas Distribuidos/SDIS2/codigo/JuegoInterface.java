import java.rmi.Remote;
import java.rmi.RemoteException;

public interface JuegoInterface extends Remote {
    Personaje newPlay(String jugador, String rol, VistaCB cb)
            throws RemoteException, BadPersonajeException;
    boolean quitPlay(VistaCB cb) throws RemoteException;
}
