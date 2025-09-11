public class Ordena1 {

    // Variables globales para contar comparaciones y asignaciones
    private static long comparaciones = 0;
    private static long asignaciones = 0;

    public static void ordena1(int[] v, int tam) {
        int i = 1, j = 2;
        int temp;
        
        while (i < tam) {
            // Comparación de elementos del vector
            comparaciones++;
            if (v[i - 1] <= v[i]) {
                i = j;
                j = j + 1;
            } else {
                // Asignaciones
                asignaciones++;
                temp = v[i - 1];
                asignaciones++;
                v[i - 1] = v[i];
                asignaciones++;
                v[i] = temp;
                asignaciones++;

                i = i - 1;
                comparaciones++;
                if (i == 0) {
                    i = 1;
                }
            }
        }
    }

    // Método para reiniciar el contador de comparaciones y asignaciones
    public static void resetContadores() {
        comparaciones = 0;
        asignaciones = 0;
    }

    // Getters para obtener los valores de las comparaciones y asignaciones
    public static long getComparaciones() {
        return comparaciones;
    }

    public static long getAsignaciones() {
        return asignaciones;
    }
}
