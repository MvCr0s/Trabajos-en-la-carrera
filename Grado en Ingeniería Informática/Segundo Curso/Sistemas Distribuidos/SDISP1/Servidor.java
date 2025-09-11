import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

public class Servidor implements Runnable {
    private int PUERTO;
    private Nodo2 nodoPadre; // Variable para almacenar el nodo padre
    protected List<Nodo2> nodosRegistrados;

    public Servidor(int puerto) {
        this.PUERTO = puerto;
        this.nodosRegistrados = new ArrayList<>();
    }

    // Método principal para iniciar el servidor
    public void iniciar() {
        try (ServerSocket serverSocket = new ServerSocket(PUERTO)) {
            System.out.println("Servidor iniciado en el puerto " + PUERTO + ". Esperando conexiones...");

            // Ciclo infinito para aceptar conexiones entrantes
            while (true) {

                Socket socket = serverSocket.accept(); // Acepta la conexión entrante
                System.out.println("Llega mensaje del cliente: " + socket.getInetAddress().getHostAddress());

                BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
        
                String inputLine;
                while ((inputLine = in.readLine()) != null) {
                    String[] partes = inputLine.split(" ");
        
                    if (partes.length == 4) {
                        // Lógica para procesar el Nodo hijo "miNombre puertoPropio hostPadre puertoPadre"
                        out.println("Mensaje recibido: " + partes[0] + " " + partes[1] + " " + partes[2] + " " + partes[3]);

                        // Creamos el sirviente para crear el nodo con los datos
                        Sirviente sirviente = new Sirviente(socket, inputLine, nodoPadre,this);  
                        Thread thread = new Thread(sirviente);
                        thread.start(); // Inicia el hilo para el sirviente 
                    } else if (partes.length == 2) {
                        // Lógica para procesar el nodo principal "Raiz puertoPropio"
                        out.println("Mensaje recibido: " + partes[0] + " " + partes[1]);

                        // Creamos el sirviente para manejar la conexión entrante
                        Sirviente sirviente = new Sirviente(socket, inputLine, nodoPadre,this);  
                        Thread thread = new Thread(sirviente);
                        thread.start(); // Inicia el hilo para el sirviente 
                    } else {
                        out.println("Formato no reconocido.");
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error en el servidor: " + e.getMessage());
        }
    }

    // Método para establecer el nodo padre
    public void setNodoPadre(Nodo2 nodoPadre) {
        this.nodoPadre = nodoPadre;
    }

    @Override
    public void run() {
        // Método iniciar() implementado aquí
        iniciar();
    }

    public synchronized void registrarNodo(Nodo2 nuevoNodo) {
        nodosRegistrados.add(nuevoNodo);
        System.out.println("Nodo registrado: "); // Asumiendo que Nodo2 tiene un método getNombre()
    }

    public Nodo2 buscarNodoPorPuerto(int puertoBuscado) {
        for (Nodo2 nodo : nodosRegistrados) {
            if (nodo.getPuerto() == puertoBuscado) {
                return nodo;
            }
        }
        return null; // Retornar null si no se encuentra el nodo con el puerto especificado
    }
}
