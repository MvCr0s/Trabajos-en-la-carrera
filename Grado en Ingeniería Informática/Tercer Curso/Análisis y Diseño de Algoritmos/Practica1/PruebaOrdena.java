
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
public class PruebaOrdena {

    public static void main(String[] args) {
        // Tamaños de vectores a probar
        int[] tamanos = {10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};
        int repeticiones = 20;  // Número de veces que se ejecuta cada prueba

         // Abrir archivo para escribir resultados
         try (BufferedWriter writer = new BufferedWriter(new FileWriter("resultados_ordenacion111.csv"))) {
            // Escribir encabezado del archivo CSV
            writer.write("Tamaño,Tiempo Promedio (ns),Comparaciones Promedio,Asignaciones Promedio\n");

            for (int tam : tamanos) {
                long tiempoTotal = 0;
                long comparacionesTotal = 0;
                long asignacionesTotal = 0;

                // Ejecutar 20 veces para cada tamaño
                for (int k = 0; k < repeticiones; k++) {
                    int[] v = new int[tam];
                    FisherYates.aleatorio(v, tam);  // Generar vector aleatorio

                    Ordena1.resetContadores();  // Reiniciar contadores

                    long inicio = System.nanoTime();  // Empezar a contar el tiempo
                    Ordena1.ordena1(v, tam);  // Ejecutar el algoritmo
                    long fin = System.nanoTime();  // Tiempo de finalización

                    tiempoTotal += (fin - inicio);  // Acumular el tiempo
                    comparacionesTotal += Ordena1.getComparaciones();  // Acumular comparaciones
                    asignacionesTotal += Ordena1.getAsignaciones();  // Acumular asignaciones
                }

                // Calcular promedios
                long tiempoPromedio = tiempoTotal / repeticiones;
                long comparacionesPromedio = comparacionesTotal / repeticiones;
                long asignacionesPromedio = asignacionesTotal / repeticiones;

                // Escribir los resultados en formato CSV (separados por comas)
                writer.write(tam + "," + tiempoPromedio + "," + comparacionesPromedio + "," + asignacionesPromedio + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}