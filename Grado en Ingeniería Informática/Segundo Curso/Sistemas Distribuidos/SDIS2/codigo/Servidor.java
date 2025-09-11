import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.rmi.RemoteException;
import java.util.ArrayList;
import java.util.List;

public class Servidor {
    private static List<PrintWriter> clientWriters = new ArrayList<>();

    public static void main(String[] args) throws RemoteException {
        // Configuración del servidor y inicialización del juego
        Juego juego = new Juego();
        int port = 10991; // Puerto para la comunicación con clientes

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("\nServidor iniciado en el puerto " + port + ". Esperando conexiones de clientes...\n");
            while (true) {
                Socket clientSocket = serverSocket.accept();
                PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                synchronized (clientWriters) {
                    clientWriters.add(out);
                }
                new Thread(new ClienteHandler(clientSocket, juego, out)).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void removeClient(PrintWriter out) {
        synchronized (clientWriters) {
            clientWriters.remove(out);
        }
    }
}

class ClienteHandler implements Runnable {
    private Socket clientSocket;
    private Juego juego;
    private PrintWriter out;

    public ClienteHandler(Socket clientSocket, Juego juego, PrintWriter out) {
        this.clientSocket = clientSocket;
        this.juego = juego;
        this.out = out;
    }

    @Override
    public void run() {
        try (
                BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
        ) {
            String userInfo = in.readLine(); // Leer información del usuario
            if (userInfo == null || userInfo.trim().isEmpty()) {
                out.println("Error: Información de usuario no proporcionada.");
                return;
            }

            String[] userArgs = userInfo.trim().split("\\s+");
            if (userArgs.length < 2) {
                out.println("Error: Formato incorrecto. Se esperaban dos argumentos: nombre y rol.");
                return;
            }

            String user = userArgs[0];
            String avatar = userArgs[1];

            // Imprimir en la consola del servidor el nombre del cliente que se ha conectado
            System.out.println("Cliente conectado: " + user + " con el avatar: " + avatar + "\n");

            // Crear el personaje para este cliente
            PersonajeImpl personaje = new PersonajeImpl(user, avatar);
            personaje.init(juego);

            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                String response = handleInput(inputLine, personaje);
                out.println(response);
            }
        } catch (SocketException e) {
            // No hace nada, de otra forma saca textazo de error
        } catch (IOException e) {
            e.printStackTrace();
        } catch (BadPersonajeException e) {
            e.printStackTrace();
        } finally {
            try {
                clientSocket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            Servidor.removeClient(out);
            System.out.println("\nCliente desconectado\n");
        }
    }

    private String handleInput(String input, PersonajeImpl personaje) throws RemoteException {
        String[] parts = input.trim().split("\\s+", 2);
        String command = parts[0];
        String argument = parts.length > 1 ? parts[1] : "";


        switch (command) {
            case "Query": case "Ask": case "Show":
                try {
                    return personaje.actua(new Accion("Query", argument));
                } catch (RemoteException e) {
                    e.printStackTrace();
                } catch (BadActionException e) {
                    e.printStackTrace();
                }
            case "Activate": case "Open": case "Disclose":
                try {
                    return personaje.actua(new Accion("Open", argument));
                } catch (RemoteException e) {
                    e.printStackTrace();
                } catch (BadActionException e) {
                    e.printStackTrace();
                }
            case "Exit": case "Abandon": case "Surrender":
                return "final.!";
            case "Help":
                return cargarMensajeAyuda();
            default:
                return "No entiendo.";
        }
    }

    private String cargarMensajeAyuda() {
        String path = "out\\production\\P2\\MensajeAyuda.txt";  // Asumiendo que resources está al mismo nivel que classes
        StringBuilder ayuda = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String linea;
            while ((linea = reader.readLine()) != null) {
                ayuda.append(linea).append("\n");
            }
        } catch (IOException e) {
            System.out.println("Error al cargar el archivo de ayuda desde " + path);
            return "No se pudo cargar el archivo de ayuda.";
        }
        return ayuda.toString();
    }
    
}

