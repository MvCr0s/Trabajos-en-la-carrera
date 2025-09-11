/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.modelos;

import uva.dis.persistencia.ProductoDTO;

/**
 *
 * @author dediego
 */
public class Producto {

    private int id;
    private String nombre;
    private double medida;
    private String unidad;
    private String descripcion;
    private double precio;

    public Producto(int id, String nombre, double medida, String unidad, String descripcion, double precio) {
        this.id = id;
        this.nombre = nombre;
        this.medida = medida;
        this.unidad = unidad;
        this.descripcion = descripcion;
        this.precio = precio;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public double getMedida() {
        return medida;
    }

    public void setMedida(double medida) {
        this.medida = medida;
    }

    public String getUnidad() {
        return unidad;
    }

    public void setUnidad(String unidad) {
        this.unidad = unidad;
    }

    public String getDescripcion() {
        return descripcion;
    }

    public void setDescripcion(String descripcion) {
        this.descripcion = descripcion;
    }

    public double getPrecio() {
        return precio;
    }

    public void setPrecio(double precio) {
        this.precio = precio;
    }

    @Override
    public String toString() {
        return nombre + " - " + medida + " " + unidad + " - " + precio + "€";
    }

    public ProductoDTO toDTO(int cantidad) {
        return new ProductoDTO(this.nombre, this.medida, this.unidad, cantidad);
    }

}
