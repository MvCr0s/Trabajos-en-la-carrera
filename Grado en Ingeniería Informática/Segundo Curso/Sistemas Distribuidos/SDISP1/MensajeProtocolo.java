public final class MensajeProtocolo implements java.io.Serializable {
  private static final long serialVersionUID = 987654321L;
  private Primitiva primitiva ;
  private java.net.InetAddress inetAddress ;
  private int puerto ;
  // private final java.net.SocketAddress comID;
  private String nombreNodo ;
  private String tokenSeguridad ;
  private String nombreVariable ;
  private int valor ;
  private String hashmap ;

  private MensajeProtocolo(Primitiva p) {
    primitiva = p;
    inetAddress = null;
    puerto = 0 ;
    nombreNodo = null;
    tokenSeguridad = null;
    nombreVariable = null;
    valor = 0;
    hashmap = null;
  }

  public static MensajeProtocolo addOk() {
    return new MensajeProtocolo(Primitiva.ADD_OK) ;
  }

  public static MensajeProtocolo multicastOk() {
    return new MensajeProtocolo(Primitiva.MULTICAST_OK);
  }

  public static MensajeProtocolo transmissionOk() {
    return new MensajeProtocolo(Primitiva.TRANSMISSION_OK) ;
  }

  public static MensajeProtocolo returnEnd() {
    return new MensajeProtocolo(Primitiva.RETURN_END) ;
  }


  public static MensajeProtocolo nothing() {
    return new MensajeProtocolo(Primitiva.NOTHING) ;
  }

  public static MensajeProtocolo badTs() {
    return new MensajeProtocolo(Primitiva.BAD_TS) ;
  }

  public static MensajeProtocolo notUnderstand() {
    return new MensajeProtocolo(Primitiva.NOTUNDERSTAND) ;
  }

  public static MensajeProtocolo add(java.net.InetAddress ia, int puerto) {
    MensajeProtocolo mp = new MensajeProtocolo(Primitiva.ADD) ;
    mp.inetAddress = ia;
    mp.puerto = puerto;
    return mp;
  }

  public static MensajeProtocolo multicast(java.net.InetAddress ia,
                 String nvar, int valor, String ts) {
    MensajeProtocolo mp = new MensajeProtocolo(Primitiva.MULTICAST) ;
    mp.nombreVariable = new String(nvar) ;
    mp.valor = valor;
    mp.tokenSeguridad = new String(ts) ;
    return mp;
  }

  public static MensajeProtocolo transmission(java.net.InetAddress ia,
                 String nvar, int valor) {
    MensajeProtocolo mp = new MensajeProtocolo(Primitiva.TRANSMISSION) ;
    mp.nombreVariable = new String(nvar) ;
    mp.valor = valor;
    return mp;
  }

  public static MensajeProtocolo RETURN(String texto) {
    MensajeProtocolo mp = new MensajeProtocolo(Primitiva.RETURN) ;
    mp.hashmap = new String(texto) ;
    return mp;
  }

  public String toString() {  /* prettyPrinter de la clase */
    return
      "<"+this.primitiva+
      "|nn>"+nombreNodo+
      "|sa>"+inetAddress+":"+puerto+
      "|ts>"+tokenSeguridad+
      "|nv>"+nombreVariable+
      "|v>"+valor+
      "|HM>"+hashmap+">.";
  }
}