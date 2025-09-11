public class Ordena2 {

    // Variables globales para contar comparaciones y asignaciones
    private static int comparaciones = 0;
    private static int asignaciones = 0;

    public static void ordena2(int[] v, int tam) {
        int k;
        int n = tam;
        for (k = n / 2; k >= 1; k--) {
            f(v, k, n);
        }
        k = n;
        while (k > 1) {
            g(v, 1, k--);
            f(v, 1, k);
        }
    }

    private static void f(int[] v, int k, int n) {
        while (2 * k <= n) {
            int j = 2 * k;

            // Comparación entre los dos hijos
            comparaciones++;
            if (j < n && v[j - 1] < v[j]) {
                j++;
            }

            // Comparación entre padre e hijo
            comparaciones++;
            if (v[k - 1] >= v[j - 1]) {
                break;
            }

            // Si el hijo es mayor, intercambiar
            g(v, k, j);
            k = j;
        }
    }

    private static void g(int[] v, int i, int j) {
        asignaciones++;
        int temp = v[i - 1];

        // Asignaciones de intercambio
        v[i - 1] = v[j - 1];
        asignaciones++;
        v[j - 1] = temp;
        asignaciones++;
    }
    // Método para reiniciar el contador de comparaciones y asignaciones
    public static void resetContadores() {
        comparaciones = 0;
        asignaciones = 0;
    }

    // Getters para obtener los valores de las comparaciones y asignaciones
    public static int getComparaciones() {
        return comparaciones;
    }

    public static int getAsignaciones() {
        return asignaciones;
    }
}
