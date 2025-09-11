/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.persistencia;

public class ProductoDTO {

    private String nombreProducto;
    private double medida;
    private String unidad;
    private int cantidad;

    public ProductoDTO(String nombreProducto, double medida, String unidad, int cantidad) {
        this.nombreProducto = nombreProducto;
        this.medida = medida;
        this.unidad = unidad;
        this.cantidad = cantidad;
    }

    public String getNombreProducto() {
        return nombreProducto;
    }

    public double getMedida() {
        return medida;
    }

    public String getUnidad() {
        return unidad;
    }

    public int getCantidad() {
        return cantidad;
    }

    @Override
    public String toString() {
        return "Producto:\n" + getNombreProducto()
                + " (" + getCantidad()
                + " " + getUnidad() + ")\n";
    }


}
