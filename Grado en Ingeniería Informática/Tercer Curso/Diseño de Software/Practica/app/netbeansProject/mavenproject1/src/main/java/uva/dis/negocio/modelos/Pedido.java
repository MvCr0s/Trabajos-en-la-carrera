/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.modelos;

import java.time.LocalDate;
import java.time.LocalTime;
import java.util.List;
import uva.dis.persistencia.ProductoDTO;

/**
 *
 * @author dediego
 */
public class Pedido {
    private String id;
    private LocalDate fecha;
    private LocalTime hora;
    private EstadoPedido estado;
    private List<ProductoDTO> productos; // nuevo campo

    public Pedido() {}

    public Pedido(String id, LocalDate fecha, LocalTime hora, EstadoPedido estado) {
        this.id = id;
        this.fecha = fecha;
        this.hora = hora;
        this.estado = estado;
    }

    // Getters
    public String getId() {
        return id;
    }

    public LocalDate getFecha() {
        return fecha;
    }

    public LocalTime getHora() {
        return hora;
    }

    public EstadoPedido getEstado() {
        return estado;
    }

    // Setters
    public void setId(String id) {
        this.id = id;
    }

    public void setFecha(LocalDate fecha) {
        this.fecha = fecha;
    }

    public void setHora(LocalTime hora) {
        this.hora = hora;
    }

    public void setEstado(EstadoPedido estado) {
        this.estado = estado;
    }
    
    public List<ProductoDTO> getProductos() {
        return productos;
    }

    public void setProductos(List<ProductoDTO> productos) {
        this.productos = productos;
    }

    @Override
    public String toString() {
        return id + " - " + hora; 
    }
}