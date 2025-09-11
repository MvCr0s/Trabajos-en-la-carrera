import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Scanner;

public class Cliente {
    public static void main(String[] args) {
        try {
            System.out.println("Conectando al servidor RMI...");
            Registry registry = LocateRegistry.getRegistry("localhost", 10991);
            JuegoInterface juego = (JuegoInterface) registry.lookup("Juego");
            VistaCBImpl callback = new VistaCBImpl();
            Scanner sc = new Scanner(System.in);

            System.out.print("Ingrese su nombre: ");
            String user = sc.nextLine();
            System.out.print("Ingrese su rol (Fighter, Wizard, Thief): ");
            String rol = sc.nextLine();

            System.out.println("Iniciando nueva partida...");
            Personaje personaje = juego.newPlay(user, rol, callback);
            imprimirBienvenida();
            System.out.println("Bienvenido al juego, " + user + " el " + rol);

            while (true) {
                System.out.print("> ");
                String input = sc.nextLine();
                String[] parts = input.split(" ", 2);
                String command = parts[0];
                String argument = parts.length > 1 ? parts[1] : "";

                try {
                    switch (command) {
                        case "Query":
                        case "Ask":
                        case "Show":
                            System.out.println(personaje.actua(new Accion("Query", argument)));
                            break;
                        case "Activate":
                        case "Open":
                        case "Disclose":
                            System.out.println(personaje.actua(new Accion("Open", argument)));
                            break;
                        case "Exit":
                        case "Abandon":
                        case "Surrender":
                            juego.quitPlay(callback);
                            System.out.println("Saliendo del juego...");
                            System.exit(0);
                            break;
                        case "Help":
                            imprimirAyuda();
                            break;
                        default:
                            System.out.println("Comando no reconocido.");
                    }
                } catch (BadActionException e) {
                    System.out.println("Acción incorrecta");
                }
            }
        } catch (Exception e) {
            System.err.println("Error en el cliente: " + e.toString());
            e.printStackTrace();
        }
    }

    private static void imprimirAyuda() {
        String nombreArchivo = "codigo\\MensajeAyuda.txt";
        try (BufferedReader reader = new BufferedReader(new FileReader(nombreArchivo))) {
            String linea;
            while ((linea = reader.readLine()) != null) {
                System.out.println(linea);
            }
        } catch (FileNotFoundException e) {
            System.err.println("Archivo de ayuda no encontrado: " + nombreArchivo);
        } catch (IOException e) {
            System.err.println("Error al leer el archivo de ayuda: " + e.getMessage());
        }
    }

    private static void imprimirBienvenida() {
        String nombreArchivo = "out\\production\\P2\\BannerBienvenida.txt";
        try (BufferedReader reader = new BufferedReader(new FileReader(nombreArchivo))) {
            String linea;
            while ((linea = reader.readLine()) != null) {
                System.out.println(linea);
            }
        } catch (FileNotFoundException e) {
            System.err.println("Archivo de bienvenida no encontrado: " + nombreArchivo);
        } catch (IOException e) {
            System.err.println("Error al leer el archivo de bienvenida: " + e.getMessage());
        }
    }
}
