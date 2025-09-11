//Controla la relacion entre cliente y servidor y el intercambio de mensajes

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class Sirviente implements Runnable{
    private final Socket socket;
    private String inputLine;
    private final Nodo2 nodoPadre;
    private Servidor servidor;


    public Sirviente(Socket socket,String inputLine, Nodo2 nodoPadre, Servidor servidor) {
        this.socket = socket;
        this.inputLine=inputLine;
        this.nodoPadre=nodoPadre;
        this.servidor = servidor;
    }

    @Override
    public void run() { 
        try {
            //BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);


                System.out.println("Mensaje recibido del cliente: " + inputLine);
                // Enviar el comando al nodo asociado para su procesamiento

                String[] partes = inputLine.split(" ");

                if (partes.length == 4) {
                    // Lógica para procesar el mensaje "miNombre puertoPropio hostPadre puertoPadre"                   
                    Nodo2 nodoPadre = servidor.buscarNodoPorPuerto(Integer.parseInt(partes[3]));
                    Nodo2 nodoHijo = new Nodo2(partes[0],Integer.parseInt(partes[1]),nodoPadre);
                    servidor.registrarNodo(nodoHijo);
                    Nodo2.agregarNodo(nodoHijo);
                    nodoPadre.addNodoHijo(nodoHijo);
                } else if (partes.length == 2) {
                    // Lógica para procesar el mensaje "Raiz puertoPropio"
                    Nodo2 nodoRaiz = new Nodo2(partes[0], Integer.parseInt(partes[1]),null);
                    Nodo2.agregarNodo(nodoRaiz);
                    servidor.registrarNodo(nodoRaiz);
                    System.out.println("Se crea el nodo " + partes[0] );
                    //procesarComando(inputLine, out);
                }

                Nodo2.imprimirlistaNodos();

            ///in.close();
            //out.close();
            //socket.close();
        } catch (IOException e) {
            System.err.println("Error al manejar la conexión: " + e.getMessage());
        }
}

}




