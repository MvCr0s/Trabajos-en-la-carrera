import java.io.*;
import java.util.*;

public class Piles {

    static class Container {
        int id, weight, resistance;

        Container(int id, int weight, int resistance) {
            this.id = id;
            this.weight = weight;
            this.resistance = resistance;
        }
    }

    // Pile is now a linked list-like structure for representing each pile
    static class Pile {
        Container first;
        Pile next;
        int totalWeight;

        Pile(Container first, Pile next, int totalWeight) {
            this.first = first;
            this.next = next;
            this.totalWeight = totalWeight;
        }
    }

    // This method returns the largest possible pile
    public static List<Container> findBestPile(List<Container> containers) {
        int n = containers.size();
        Pile[] piles = new Pile[n + 1];  // An array to store piles for each height
        piles[0] = new Pile(null, null, 0);  // The empty pile (height = 0)
        int bestHeight = 0;

        // Loop through each container, starting from the last one
        for (int i = n - 1; i >= 0; i--) {
            Container newContainer = containers.get(i);
            for (int h = bestHeight; h >= 0; h--) {
                // Check if the container can be added to the pile of height h
                if (newContainer.resistance >= piles[h].totalWeight) {
                    // Create a new pile by adding this container on top of pile[h]
                    Pile newPile = new Pile(newContainer, piles[h], newContainer.weight + piles[h].totalWeight);

                    // Update the pile array
                    if (h == bestHeight) {
                        piles[h + 1] = newPile;
                        bestHeight = h + 1;
                    } else if (newPile.totalWeight < piles[h + 1].totalWeight) {
                        piles[h + 1] = newPile;
                    }
                }
            }
        }

        // Reconstruct the best pile from the bestHeight pile
        List<Container> bestPile = new ArrayList<>();
        Pile currentPile = piles[bestHeight];
        while (currentPile != null && currentPile.first != null) {
            bestPile.add(currentPile.first);
            currentPile = currentPile.next;
        }

        // The pile is already in the correct order (from base to top),
        // so no need to reverse it
        return bestPile;
    }

    public static void main(String[] args) {
        List<Container> containers = new ArrayList<>();
        // Leer archivo y cargar datos
        try (BufferedReader reader = new BufferedReader(new FileReader("entrada5.txt"))) {
            int numeroContenedores = Integer.parseInt(reader.readLine());
            for (int i = 0; i < numeroContenedores; i++) {
                String[] partes = reader.readLine().split(" ");
                int weight = Integer.parseInt(partes[0]);
                int resistance = Integer.parseInt(partes[1]);
                containers.add(new Container(i + 1, weight, resistance));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Medir el tiempo que tarda en encontrar la mejor pila
        long startTime = System.currentTimeMillis();  // Captura el tiempo antes de la operación

        // Encontrar la mejor pila posible
        List<Container> bestPile = findBestPile(containers);

        long endTime = System.currentTimeMillis();  // Captura el tiempo después de la operación

        // Calcular la duración en segundos
        double durationInSeconds = (endTime - startTime) / 1000.0;  // Convertir milisegundos a segundos

        // Imprimir el tiempo de ejecución en segundos
        System.out.println("Tiempo de ejecución: " + durationInSeconds + " segundos");

        // Imprimir el resultado en el orden de base a la parte superior (sin invertir)
        System.out.println("Número de contenedores en la mejor pila: " + bestPile.size());
        for (Container container : bestPile) {
            System.out.println("Contenedor ID: " + container.id);
        }
    }
}
