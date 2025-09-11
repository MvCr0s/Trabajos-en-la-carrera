import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.RemoteException;

public class Lanzador {
    public static void main(String[] args) {
        try {
            // Crear una instancia del juego
            Juego juego = new Juego();

            // Crear el registro en el puerto 1099
            Registry registry = LocateRegistry.createRegistry(10991);

            // Registrar el objeto del juego en el registro
            registry.rebind("Juego", juego);

            System.out.println("Servidor del juego iniciado y listo para aceptar conexiones.");
        } catch (RemoteException e) {
            System.out.println("Excepción del servidor: " + e.toString());
            e.printStackTrace();
        }
    }
}
