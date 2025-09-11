import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.List;
import java.util.Set;


public class ClienteMulticast {
    


    public static void enviarMensaje(String hostName, int portNumber, String varName, int varValue) throws UnknownHostException, IOException {
        // Formato del mensaje a enviar al nodo
        String mensaje = "M:" + varName + ":" + varValue;
        
        // Establecer conexión con el nodo
        try (Socket socket = new Socket(hostName, portNumber);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
            // Enviar mensaje al nodo
            out.println(mensaje);
            System.out.println("Mensaje enviado al nodo en el puerto " + portNumber + ": " + mensaje);

            // Leer la respuesta del nodo
            String respuesta = in.readLine();
            System.out.println("Respuesta recibida del nodo en el puerto " + portNumber + ": " + respuesta);
        } catch (IOException e) {
            System.err.println("Error al enviar o recibir mensaje del nodo en el puerto " + portNumber + ": " + e.getMessage());
        }
    }

    
    public static void main(String[] args) {
        try (BufferedReader stdIn = new BufferedReader(new InputStreamReader(System.in))) {
            //Mensajes Bienvenida Cliente
            System.out.println("ClienteMulticast iniciado. Puede comenzar a enviar comandos al servidor.");
            System.out.println("Escriba 'exit' para salir.");

            while (true) {
                String userInput = stdIn.readLine();

                if ("exit".equalsIgnoreCase(userInput)) {
                    System.out.println("Saliendo del cliente...");
                    break;
                }
                String[] partes = userInput.split(" ");
                if (partes.length != 4) {
                    System.err.println("Uso: <hostname> <puerto> <nombreVariable> <valor>");
                    System.exit(1);
                }
        
                String hostName = partes[0];
                int portNumber = Integer.parseInt(partes[1]);
                String varName = partes[2];
                int varValue = Integer.parseInt(partes[3]);
            
                enviarMensaje(hostName, portNumber, varName, varValue);
            }
    }catch (IOException e) {
        System.err.println("Error de entrada/salida: " + e.getMessage());
    }
}

}
