import java.util.Random;

public class FisherYates {

    // Método para generar un vector aleatorio con el algoritmo de Fisher-Yates
    public static void aleatorio(int[] v, int tam) {
        Random random = new Random();
        for (int i = 0; i < tam; i++) {
            v[i] = i;
        }

        for (int i = tam - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = v[i];
            v[i] = v[j];
            v[j] = temp;
        }
    }
}
