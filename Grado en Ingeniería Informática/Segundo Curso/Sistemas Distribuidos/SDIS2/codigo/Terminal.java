


import java.io.*;
import java.rmi.RemoteException;

public class Terminal {
  public static void main(String [] args) throws RemoteException {
    java.util.Scanner sc = new java.util.Scanner(System.in);

    java.io.BufferedReader br;
    String linea;
    try {
      // Imprime banner de inicio
      br = new java.io.BufferedReader(
	     new java.io.FileReader("BannerBienvenida.txt"));
      while ((linea = br.readLine()) != null)
        System.out.println(linea);
    } catch (java.io.FileNotFoundException fnfe) {
      System.err.println("Terminal: Banner file not found!\n");
    } catch (java.io.IOException ioe) {
      System.err.println("Terminal: Error reading banner!\n");
    } finally {
      System.out.println("Welcome to MULTIMAP!\n\n");
    }

    // Captura de la ayuda
    String ayuda="";
    try {
      br = new java.io.BufferedReader(
	     new java.io.FileReader("MensajeAyuda.txt"));
      while ((linea = br.readLine()) != null)
        ayuda += linea+"\n";
    } catch (java.io.FileNotFoundException fnfe) {
      System.err.println("Terminal: Help file not found!\n");
      ayuda = "This is all the help you will get...\n";
    } catch (java.io.IOException ioe) {
      System.err.println("Terminal: Error reading help file!\n");
      ayuda = "This is all the help you will get...\n";
    } 

    System.out.println(ayuda+"\n");

    String user   = (args.length > 0) ? args[0] : "Anonimo";
    String avatar = (args.length > 1) ? args[1] : "Wizard";

    if (!avatar.equals("Wizard") && !avatar.equals("Fighter")
        && !avatar.equals("Thief")) {
      System.out.println("Terminal: Avatar inexistente: "+avatar);
      System.exit(1);
    }

    System.out.println("Hello ["+user+"] ["+avatar+"]");
    /* **
     * Primero creamos un juego
     */
    Juego juego = new Juego();

    try {
      // esta parte es propia de un servidor que conoce la
      // implementacion del gestor de personajes (aka PersonajeImpl)
      PersonajeImpl gestorPers = new PersonajeImpl(user, avatar);
      gestorPers.init(juego);
  
      // esta parte es la interfaz del gestor mas propia de un
      // cliente remoto.
      Personaje miPersonaje = (Personaje) gestorPers;
  
      java.util.Map<String,String> info = miPersonaje.info();
      System.out.println(info);
  
      System.out.println("--------------------------");
  
      // lista de acciones posibles para mi personaje en cada
      // momento del juego.
      java.util.List<Accion> acciones;
  
      String orden ;
      String argumento ;
      while(true) {
        acciones = miPersonaje.acciones();
   System.out.println("----------ACCIONES-----------");
        System.out.println(acciones);
   System.out.println("--------------------------");
        System.out.print("> ");
        orden = sc.next();
        switch (orden) {
          case "Query": case "Ask": case "Show":
            try {
              argumento = sc.next();
              if (null != argumento) {
		Accion acc = new Accion("Query", argumento);
                System.out.println(acc);
                System.out.println(miPersonaje.actua(acc));
	      }
	    } catch (java.util.NoSuchElementException nsee) {}
            break;
          case "Activate": case "Open": case "Disclose":
            try {
              argumento = sc.next();
              if (null != argumento) {
		Accion acc = new Accion("Open", argumento);
                System.out.println(acc);
                System.out.println(miPersonaje.actua(acc));
	      }
	    } catch (java.util.NoSuchElementException nsee) {}
            break;
          case "Exit": case "Abandon": case "Surrender":
            System.out.println("final.!");
            System.exit(0);
            break;
          case "Help":
	    System.out.println(ayuda);
	    break;
          default:
            System.out.println("No entiendo.");
        }
      }
    } catch (BadPersonajeException bpe) {
      System.err.println("Terminal: BadPersonajeException.");
    } catch (BadActionException bae) {
      System.err.println("Terminal: BadActionException.");
    }
  }
}
