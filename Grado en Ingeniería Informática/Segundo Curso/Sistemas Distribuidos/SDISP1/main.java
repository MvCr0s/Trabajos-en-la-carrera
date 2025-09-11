public class main {
    public static void main(String[] args) {
        Runnable servidor = new Servidor(8080);
        new Thread(servidor, "Hilo Servidor").start();
    }
}
