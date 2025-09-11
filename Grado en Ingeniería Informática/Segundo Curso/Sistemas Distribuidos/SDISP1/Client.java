//El cliente que se conectará al servidor

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class Client {
    public static void main(String[] args) {
        try (BufferedReader stdIn = new BufferedReader(new InputStreamReader(System.in))) {
            //Mensajes Bienvenida Cliente
            System.out.println("Cliente iniciado. Puede comenzar a enviar comandos al servidor.");

            System.out.println("- Para agregar un nodo hijo al Servidor:  miNombre puertoPropio hostPadre puertoPadre");
            System.out.println("- Para añadir el nodo principal Raiz puertoPropio");
                                                               

            System.out.println("Escriba 'exit' para salir.");

            while (true) {
                String userInput = stdIn.readLine();

                if ("exit".equalsIgnoreCase(userInput)) {
                    System.out.println("Saliendo del cliente...");
                    break;
                }

                try (Socket socket = new Socket("localhost",8080); // Cambia "localhost" y el puerto según la config --> Esto posiblemente haya que cambiarlo
                     PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                     BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

                    out.println(userInput); // Envía los datos al servidor

                    // Esperar y mostrar la respuesta del servidor
                    System.out.println("Esperando respuesta del servidor...");
                    String respuesta = in.readLine();
                    System.out.println("Respuesta del servidor: " + respuesta);

                } catch (IOException e) {
                    System.err.println("Error al intentar conectarse al servidor: " + e.getMessage());
                }
                
            }
        } catch (IOException e) {
            System.err.println("Error de entrada/salida: " + e.getMessage());
        }
    }
}
