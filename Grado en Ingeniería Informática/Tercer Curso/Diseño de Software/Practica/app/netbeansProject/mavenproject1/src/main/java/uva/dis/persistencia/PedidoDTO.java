/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.persistencia;

import java.time.LocalDate;
import java.time.LocalTime;
import java.util.List;
import uva.dis.negocio.modelos.EstadoPedido;

/**
 *
 * @author dediego
 */
public class PedidoDTO {

    private String id;
    private LocalDate fecha;
    private LocalTime hora;
    private EstadoPedido estado;
    private List<ProductoDTO> productos;

    public PedidoDTO(String id, LocalDate fecha, LocalTime hora, EstadoPedido estado) {
        this.id = id;
        this.fecha = fecha;
        this.hora = hora;
        this.estado = estado;
    }

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
    
    public List<ProductoDTO> getProductos() {
        return productos;
    }

    public void setProductos(List<ProductoDTO> productos) {
        this.productos = productos;
    }
    
    public String toString2() {
        return     " ID Pedido: " + getId() + "\n" +
    " Fecha: " + getFecha() + "\n" +
    " Hora: " + getHora() + "\n" +
    " Estado: " + getEstado() + "\n" ;
    }


    @Override
    public String toString() {
        return id + " - " + hora; 
    }
}
