import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class VistaCBImpl extends UnicastRemoteObject implements VistaCB {
    protected VistaCBImpl() throws RemoteException {
        super();
    }

    @Override
    public String updateView(String message) throws RemoteException {
        System.out.println("Mensaje del servidor: " + message);
        return "Mensaje recibido";
    }
}
