public class Ordena3 {

    // Variables globales para contar comparaciones y asignaciones
    private static int comparaciones = 0;
    private static int asignaciones = 0;

    public static void ordena3(int[] v, int tam) {
        int m = h(v, tam);
        int[] c = new int[m + 1];
        int[] w = new int[tam];
        for (int i = 0; i < tam; i++) {
            asignaciones++;
            c[v[i]] = c[v[i]] + 1;
        }
        for (int i = 1; i < m + 1; i++) {
            c[i] = c[i] + c[i - 1];
        }
        for (int i = tam - 1; i >= 0; i--) {
            asignaciones++;
            w[c[v[i]] - 1] = v[i];
            asignaciones++;
            c[v[i]] = c[v[i]] - 1;
        }
        for (int i = 0; i < tam; i++) {
            asignaciones++;
            v[i] = w[i];
        }
    }

    private static int h(int[] v, int tam) {
        int i;
        asignaciones++;
        int m = v[0];
        for (i = 1; i < tam; i++) {
            comparaciones++;
            if (v[i] > m) {
                asignaciones++;
                m = v[i];
            }
        }
        return m;
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
