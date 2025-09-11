import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class PruebaOrdena2 {

    public static void main(String[] args) {
        // Tamaños de vectores a probar
        int[] tamanos = {
            10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
            150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000,
            550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000, 950000, 1000000,
            1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000,
            2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000, 3000000,
            3100000, 3200000, 3300000, 3400000, 3500000, 3600000, 3700000, 3800000, 3900000, 4000000,
            4100000, 4200000, 4300000, 4400000, 4500000, 4600000, 4700000, 4800000, 4900000, 5000000
        };
                int repeticiones = 20;  // Número de veces que se ejecuta cada prueba

        // Abrir archivo para escribir resultados
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("resultados_ordenacion2.csv"))) {
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

                    Ordena2.resetContadores();  // Reiniciar contadores

                    long inicio = System.nanoTime();  // Empezar a contar el tiempo
                    Ordena2.ordena2(v, tam);  // Ejecutar el algoritmo
                    long fin = System.nanoTime();  // Tiempo de finalización

                    tiempoTotal += (fin - inicio);  // Acumular el tiempo
                    comparacionesTotal += Ordena2.getComparaciones();  // Acumular comparaciones
                    asignacionesTotal += Ordena2.getAsignaciones();  // Acumular asignaciones
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
