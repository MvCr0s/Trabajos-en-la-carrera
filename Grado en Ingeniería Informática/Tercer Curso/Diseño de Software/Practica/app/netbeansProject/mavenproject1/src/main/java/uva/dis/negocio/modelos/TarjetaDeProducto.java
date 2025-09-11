/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.modelos;

/**
 *
 * @author Admin
 */
public class TarjetaDeProducto {
    
    private int id;
    private String nombre;
    private Short unidad;
    private String descripcion;
    private String alergenos;
    private String ingredientes;
    private String negocio;

    public TarjetaDeProducto(int id, String nombre, Short unidad, String descripcion, String alergenos, String ingredientes, String negocio) {
        this.id = id;
        this.nombre = nombre;
        this.unidad = unidad;
        this.descripcion = descripcion;
        this.alergenos = alergenos;
        this.ingredientes = ingredientes;
        this.negocio = negocio;
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

    public Short getUnidad() {
        return unidad;
    }

    public void setUnidad(Short unidad) {
        this.unidad = unidad;
    }

    public String getDescripcion() {
        return descripcion;
    }

    public void setDescripcion(String descripcion) {
        this.descripcion = descripcion;
    }

    public String getAlergenos() {
        return alergenos;
    }

    public void setAlergenos(String alergenos) {
        this.alergenos = alergenos;
    }

    public String getIngredientes() {
        return ingredientes;
    }

    public void setIngredientes(String ingredientes) {
        this.ingredientes = ingredientes;
    }

    public String getNegocio() {
        return negocio;
    }

    public void setNegocio(String negocio) {
        this.negocio = negocio;
    }
    
    
    
}
