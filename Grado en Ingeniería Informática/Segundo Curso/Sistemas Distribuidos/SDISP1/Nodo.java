import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Nodo implements Runnable {
    private String nombre; // Nombre del nodo
    private int puerto; // Puerto en el que el nodo escucha
    private Map<String, String> variables = new HashMap<>(); // Mapa para almacenar variables registradas
    private List<Nodo> nodosHijos = new ArrayList<>(); // Lista de nodos hijos

    public Nodo(String nombre, int puerto) {
        this.nombre = nombre; // Inicializar nombre del nodo
        this.puerto = puerto; // Inicializar puerto del nodo
    }

    public void run() {
        try (ServerSocket serverSocket = new ServerSocket(puerto)) { // Crear ServerSocket para escuchar conexiones
            System.out.println("Nodo " + nombre + " escuchando en puerto " + puerto); // Mensaje de inicio
            while (true) { // Bucle infinito para aceptar conexiones
                try (Socket clientSocket = serverSocket.accept(); // Aceptar una conexión entrante
                     PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                     BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) { // Flujos para enviar y recibir datos
                    String inputLine;
                    while ((inputLine = in.readLine()) != null) { // Leer líneas de texto del cliente
                        System.out.println("Recibido: " + inputLine); // Mostrar mensaje recibido
                        String[] tokens = inputLine.split(" "); // Dividir la línea en tokens
                        String command = tokens[0].toUpperCase(); // Obtener el comando en mayúsculas
                        switch (command) { // Procesar el comando
                            case "REGISTER": // Comando para registrar una variable
                                if (tokens.length == 3) { // Verificar el formato del comando
                                    registerVariable(tokens[1], tokens[2], out); // Registrar la variable
                                } else {
                                    out.println("ERROR Invalid REGISTER syntax"); // Enviar mensaje de error
                                }
                                break;
                            case "QUERY": // Comando para consultar una variable
                                if (tokens.length == 2) { // Verificar el formato del comando
                                    queryVariable(tokens[1], out); // Consultar la variable
                                } else {
                                    out.println("ERROR Invalid QUERY syntax"); // Enviar mensaje de error
                                }
                                break;
                            case "ADDCHILD": // Comando para agregar un nodo hijo
                                System.out.println("ADDCHILD command received"); // Mensaje de depuración
                                if (tokens.length == 3) { // Verificar el formato del comando
                                    addChild(tokens[1], Integer.parseInt(tokens[2]), out);
                                    propagateVariablesToChildren(); // Propagar las variables a los nodos hijos
                                } else {
                                    out.println("ERROR Invalid ADDCHILD syntax"); // Enviar mensaje de error
                                }
                                break;
                            case "SENDVARIABLES": // Comando para enviar variables a los nodos hijos
                                sendVariablesToChildren(out);
                                break;
                            default:
                                out.println("ERROR Unknown command"); // Enviar mensaje de error para comandos desconocidos
                                break;
                        }
                    }
                }
            }
        } catch (IOException e) { // Capturar excepciones de E/S
            System.out.println("Exception caught when trying to listen on port " + puerto + " or listening for a connection"); // Mensaje de excepción
            System.out.println(e.getMessage()); // Mostrar mensaje de excepción
        }
    }

    private void registerVariable(String varName, String varValue, PrintWriter out) {
        variables.put(varName, varValue); // Registrar la variable en el mapa
        out.println("OK Variable " + varName + " registered with value " + varValue); // Enviar confirmación al cliente
    }

    private void queryVariable(String varName, PrintWriter out) {
        String value = variables.get(varName); // Obtener el valor de la variable
        if (value != null) {
            out.println("VALUE " + varName + " " + value); // Enviar el valor de la variable al cliente
        } else {
            out.println("ERROR Variable " + varName + " not found"); // Enviar mensaje de error si la variable no se encuentra
        }
    }

    private void addChild(String nombreHijo, int puertoHijo, PrintWriter out) {
        Nodo hijo = new Nodo(nombreHijo, puertoHijo); // Crear un nuevo nodo hijo
        nodosHijos.add(hijo); // Agregar el nuevo nodo hijo a la lista de nodos hijos
        Thread thread = new Thread(hijo); // Crear un nuevo hilo para ejecutar el nodo hijo
        thread.start(); // Iniciar el hilo del nodo hijo
        out.println("OK Child node " + nombreHijo + " added"); // Enviar confirmación al cliente
    }

    private void propagateVariablesToChildren() {
        for (Nodo hijo : nodosHijos) { // Iterar sobre todos los nodos hijos
            hijo.variables.putAll(this.variables); // Propagar las variables al nodo hijo
        }
    }

    private void sendVariablesToChildren(PrintWriter out) {
        for (Nodo hijo : nodosHijos) { // Iterar sobre todos los nodos hijos
            for (Map.Entry<String, String> entry : variables.entrySet()) { // Iterar sobre las variables del nodo actual
                hijo.variables.put(entry.getKey(), entry.getValue()); // Enviar las variables al nodo hijo
            }
        }
        out.println("OK Variables sent to children nodes"); // Enviar confirmación al cliente
    }
}

