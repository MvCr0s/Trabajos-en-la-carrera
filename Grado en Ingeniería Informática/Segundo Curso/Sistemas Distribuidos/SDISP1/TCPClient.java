import java.io.*;
import java.net.Socket;

public class TCPClient {
    public static void main(String[] args) {
        String hostName = "localhost"; // Dirección IP o nombre del host del servidor
        int portNumber = 8080; // Puerto del servidor

        try (
            // Se establece una conexión TCP con el servidor en el host y puerto especificados
            Socket socket = new Socket(hostName, portNumber);
            // Se crea un flujo de salida para enviar datos al servidor
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            // Se crea un flujo de entrada para recibir datos del servidor
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            // Se crea un flujo de entrada para leer datos del usuario desde la consola
            BufferedReader stdIn = new BufferedReader(new InputStreamReader(System.in))
        ) {
            String userInput; // Variable para almacenar la entrada del usuario
            // Se lee la entrada del usuario desde la consola
            while ((userInput = stdIn.readLine()) != null) {
                // Se envía la entrada del usuario al servidor
                out.println(userInput);
                // Se lee la respuesta del servidor
                System.out.println("Echo from server: " + in.readLine());
            }
        } catch (IOException e) {
            // Se imprime un mensaje de error si no se puede establecer la conexión con el servidor
            System.err.println("No se puede conectar al servidor en " + hostName + ":" + portNumber);
            // Se sale del programa con un código de error
            System.exit(1);
        }
    }
}