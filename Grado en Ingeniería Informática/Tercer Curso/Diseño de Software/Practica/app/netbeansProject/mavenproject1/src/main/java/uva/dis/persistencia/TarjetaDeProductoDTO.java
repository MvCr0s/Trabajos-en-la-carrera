/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.persistencia;

/**
 *
 * @author dediego
 */
public class TarjetaDeProductoDTO {

    private int id; // ✅ Campo nuevo
    private String nombre;
    private short unidad;
    private String descripcion;
    private String alergenos;
    private String ingredientes;
    private String negocio;

    public TarjetaDeProductoDTO(int id, String nombre, short unidad, String descripcion, String alergenos, String ingredientes, String negocio) {
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

    public short getUnidad() {
        return unidad;
    }

    public String getDescripcion() {
        return descripcion;
    }

    public String getAlergenos() {
        return alergenos;
    }

    public String getIngredientes() {
        return ingredientes;
    }

    public String getNegocio() {
        return negocio;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public void setUnidad(short unidad) {
        this.unidad = unidad;
    }

    public void setDescripcion(String descripcion) {
        this.descripcion = descripcion;
    }

    public void setAlergenos(String alergenos) {
        this.alergenos = alergenos;
    }

    public void setIngredientes(String ingredientes) {
        this.ingredientes = ingredientes;
    }

    public void setNegocio(String negocio) {
        this.negocio = negocio;
    }
}

