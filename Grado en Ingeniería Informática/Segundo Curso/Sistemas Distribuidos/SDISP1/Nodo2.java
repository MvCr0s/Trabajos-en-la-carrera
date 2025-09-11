import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Nodo2 {
    private String nombre;
    private int puerto;
    private String tokenSeguridad;
    private Nodo2 nodoPadre; // Nuevo atributo para el nodo padre
    private Map<String, Integer> libroVariables = new HashMap<>();
    private Map<String, ArrayList<Integer>> nodosHijos = new ConcurrentHashMap<>();
    private static List<Nodo2> listaNodos = new ArrayList<>();
    private Thread hiloEscucha;


    public Nodo2() {
    }

    private void iniciarEscucha() {
        hiloEscucha = new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(puerto)) {
                System.out.println("Nodo " + nombre + " escuchando en el puerto " + puerto);
                while (true) {
                    Socket socket = serverSocket.accept();
                    // Aquí puedes manejar la conexión entrante
                    System.out.println("Se ha establecido una conexión entrante en el nodo " + nombre);
                    manejarConexion(socket);
                }
            } catch (IOException e) {
                System.err.println("Error al iniciar la escucha en el nodo " + nombre + ": " + e.getMessage());
            }
        });
        hiloEscucha.start();
    }

    public void detenerEscucha() {
        if (hiloEscucha != null && !hiloEscucha.isInterrupted()) {
            hiloEscucha.interrupt();
        }
    }


    public static synchronized void agregarNodo(Nodo2 nodo) {
        listaNodos.add(nodo);
    }
    
    public static Nodo2 buscarPorPuerto(int puerto) {
        synchronized (listaNodos) {
            for (Nodo2 nodo : listaNodos) {
                if (nodo.getPuerto() == puerto) {
                    return nodo;
                }
            }
        }
        return null;
    }

    public static synchronized List<Nodo2> getListaNodos() {
        return new ArrayList<>(listaNodos);
    }

    public static void imprimirlistaNodos() {
        synchronized (listaNodos) {
            System.out.println("Listado de todos los nodos:");
            for (Nodo2 nodo : listaNodos) {
                System.out.println(nodo);
            }
        }
    }

    // Sobrescribiendo el método toString para que devuelva una representación útil del nodo
    @Override
    public String toString() {
        return "Nodo{" +
                "nombre='" + nombre + '\'' +
                ", puerto=" + puerto +
                '}';
    }


    public Nodo2(String nombre, int puerto, Nodo2 nodoPadre) {
        this.nombre = nombre;
        this.puerto = puerto;
        this.nodoPadre = nodoPadre;
        iniciarEscucha();
    }

    // Método para agregar nodo hijo al nodo actual o al nodo padre
    public void addNodoHijo(Nodo2 hijo) {
       agregarHijo(hijo.getNombre(),hijo.getPuerto());

    }


    public void agregarHijo(String clave, int valor) {
        // Verificar si ya existe una entrada para la clave en el mapa
        if (!nodosHijos.containsKey(clave)) {
            nodosHijos.put(clave, new ArrayList<>()); // Si no existe, crear una nueva lista para esa clave
        }
        // Obtener la lista asociada con la clave y agregar el valor a esa lista
        ArrayList<Integer> listaValores = nodosHijos.get(clave);
        listaValores.add(valor);
        visualizarJerarquia();
    }
    

    public void visualizarJerarquia() {
        System.out.println("Jerarquía de nodos para el nodo " + nombre + ":");
        visualizarNodosHijosRecursivo(this, 0);
    }

    private void visualizarNodosHijosRecursivo(Nodo2 nodo, int nivel) {
        String prefix = "|--".repeat(nivel);
        System.out.println(prefix + nodo.nombre + ":" + nodo.puerto);
        for (Map.Entry<String, ArrayList<Integer>> entry : nodo.nodosHijos.entrySet()) {
            String nombreHijo = entry.getKey();
            ArrayList<Integer> puertos = entry.getValue();
            for (Integer puerto : puertos) {
                System.out.println(prefix + "   |--" + nombreHijo + ":" + puerto);
            }
        }
    }

    public Nodo2 buscarNodo(String nombre) {
        // No es necesario modificar esta función ya que devuelve un nodo por su nombre
        return nodosHijos.containsKey(nombre) ? this : null;
    }

    public String getNombre() {
        return nombre; // Se asume que el nombre del nodo es equivalente al host
    }

    public int getPuerto() {
        return puerto; // Retorna el puerto del nodo
    }

    public Map<String, Integer> getLibroVariables() {
        return libroVariables;
    }

    public Map<String, ArrayList<Integer>> getNodosHijos() {
        return nodosHijos;
    }

    /*public void recibirVariable(String mensaje) {
        String[] partes = mensaje.split(":");
        if (partes.length != 3 || !partes[0].equals("M")) {
            System.err.println("Formato de mensaje incorrecto. Se esperaba 'M:nombreVariable:valor'");
            return;
        }
        
        String nombreVar = partes[1];
        int valor = Integer.parseInt(partes[2]);
        
        libroVariables.put(nombreVar, valor);
        System.out.println("Variable " + nombreVar + " con valor " + valor + " almacenada en " + nombre);
        
        // Opcional: Propagar la variable a los nodos hijos
        for (ArrayList<Integer> puertos : nodosHijos.values()) {
            for (Integer puerto : puertos) {
                libroVariables.put(nombreVar, valor);
            }
        }
    }*/

    private void manejarConexion(Socket socket) {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {
            String mensaje = in.readLine();
            System.out.println("Mensaje recibido del cliente: " + mensaje);
    
            // Procesar mensaje y actualizar el libro de variables
            String[] partes = mensaje.split(":");
            if (partes.length == 3 && partes[0].equals("M")) {
                libroVariables.put(partes[1], Integer.parseInt(partes[2]));
                System.out.println("Actualizado LibroVar padre ");
    
                // Transformar el mensaje a X y enviarlo a los nodos hijos
                String mensajeHijos = "X:" + partes[1] + ":" + Integer.parseInt(partes[2]);
                String respuestaHijos = notificarNodosHijos(mensajeHijos);
                
                // Devolver todas las variables del libro de variables junto con las respuestas de los nodos hijos
                out.println(getVariablesFormatoMulticast() + respuestaHijos);
                System.out.println(getVariablesFormatoMulticast() + respuestaHijos);
                socket.close();
            } else if (partes.length == 3 && partes[0].equals("X")) {
                String variable = partes[1];
                int valor = Integer.parseInt(partes[2]);
    
                // Si la variable no está en el libro de variables, actualizarla y notificar a los hijos
                if (!libroVariables.containsKey(variable)) {
                    libroVariables.put(variable, valor);
                    System.out.println("Actualizado LibroVar hijo");
    
                    // Transformar el mensaje a X y enviarlo a los nodos hijos
                    String mensajeHijos = "X:" + variable + ":" + valor;
                    String respuestaHijos = notificarNodosHijos(mensajeHijos);
    
                    // Devolver todas las variables del libro de variables junto con las respuestas de los nodos hijos
                    out.println(getVariablesFormatoMulticast() + respuestaHijos);
                    System.out.println(getVariablesFormatoMulticast() + respuestaHijos);
                    socket.close();
                }
            } else {
                System.err.println("Formato de mensaje incorrecto: " + mensaje);
            }
        } catch (IOException e) {
            System.err.println("Error al manejar la conexión: " + e.getMessage());
        }
    }
    
    
    // Método para obtener todas las variables en formato multicast
    private String getVariablesFormatoMulticast() {
        StringBuilder variablesMulticast = new StringBuilder();
        variablesMulticast.append(nombre).append(": ");
    
        for (Map.Entry<String, Integer> entry : libroVariables.entrySet()) {
            String variable = entry.getKey();
            int valor = entry.getValue();
            variablesMulticast.append("{").append(variable).append("=").append(valor).append("} ");
        }
    
        return variablesMulticast.toString();
    }
    


    // Método para notificar a los nodos hijos y recopilar respuestas
private String notificarNodosHijos(String mensaje) {
    StringBuilder respuestaTotal = new StringBuilder();

    for (Map.Entry<String, ArrayList<Integer>> entry : nodosHijos.entrySet()) {
        String nombreHijo = entry.getKey();
        ArrayList<Integer> puertos = entry.getValue();

        for (Integer puerto : puertos) {
            try (Socket socket = new Socket("localhost", puerto);
                    PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                    BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
                out.println(mensaje);
                System.out.println("Mensaje enviado al nodo hijo " + nombreHijo + ": " + mensaje);

                // Leer la respuesta del nodo hijo
                String respuesta = in.readLine();
                respuestaTotal.append(respuesta).append("   ");
            } catch (IOException e) {
                System.err.println("Error al enviar mensaje al nodo hijo " + nombreHijo + ": " + e.getMessage());
            }
        }
    }

    return respuestaTotal.toString();
}

}