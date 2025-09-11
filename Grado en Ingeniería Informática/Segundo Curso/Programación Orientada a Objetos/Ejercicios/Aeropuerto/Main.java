package Aeropuerto;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        ArrayList<String> lista = new ArrayList<>();
        lista.add("A"); // índice 0
        lista.add("B"); // índice 1
        lista.add("C"); // índice 2

        System.out.println("Lista antes de insertar: " + lista);

        // Añadir "X" al final de la lista
        lista.add("X");

        System.out.println("Lista después de insertar: " + lista);
    }
}
