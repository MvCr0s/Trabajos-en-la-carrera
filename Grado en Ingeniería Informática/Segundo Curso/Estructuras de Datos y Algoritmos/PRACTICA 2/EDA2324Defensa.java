import java.util.Random;
import java.util.Scanner;

public class EDA2324Defensa {
    
    public static void main(String[] args) {
        Scanner teclado = new Scanner(System.in);
        System.out.print("Semilla = ");
        int semilla = teclado.nextInt();
        Random rnd = new Random(semilla);
        System.out.print("Nº Filas (n) = ");
        int n = teclado.nextInt();
        System.out.print("Nº Columnas (m) = ");
        int m = teclado.nextInt();
        Celda celda = new CeldaAvanzada();  
        int num_rayos = 0;
        // Simulación
        celda.Inicializar(n, m);
        while(!celda.Cortocircuito()) {
            int i = rnd.nextInt(n);
            int j = rnd.nextInt(m);
            celda.RayoCosmico(i, j);
            num_rayos++;
        }            
        System.out.println("Nº Rayos = "+num_rayos);
    }
}