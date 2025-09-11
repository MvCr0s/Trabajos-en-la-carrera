/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.interfaz.vistactrl.consultarpedido;

import java.util.List;
import javax.swing.DefaultListModel;
import uva.dis.persistencia.PedidoDTO;
import uva.dis.persistencia.ProductoDTO;
import uva.dis.negocio.modelos.Empleado;
import uva.dis.negocio.modelos.Session;
import uva.dis.interfaz.vistactrl.GestorDeVistas;
import uva.dis.negocio.controladorescasouso.ControladorCUConsultarPedidos;


/**
 *
 * @author dediego
 */
public class ControladorVistaConsultarPedido {
    
    private final ConsultarPedidoVista vista;


    public ControladorVistaConsultarPedido(ConsultarPedidoVista vista) {
        this.vista = vista;
         
    }
    

    public void obtenerPedidosSinRecoger(String filtro) {
        String idNegocio = ControladorCUConsultarPedidos.getNegocio();
        List<PedidoDTO> listaPedidos = ControladorCUConsultarPedidos.buscarPedidosSinRecoger(idNegocio, filtro);

        DefaultListModel<PedidoDTO> model = new DefaultListModel<>();

        if (listaPedidos.isEmpty()) {
            vista.mostrarError("No se ha encontrado ningun pedido sin recoger actualmente");
        } else {
            for (PedidoDTO pedido : listaPedidos) {
                model.addElement(pedido);
            }
            vista.mostrarError("");
        }

        vista.setListaPedidos(model);
    }

    public String obtenerDetallesPedido(PedidoDTO pedido) {
        StringBuilder detalles = new StringBuilder();
        detalles.append(pedido.toString2());

        for (ProductoDTO producto : pedido.getProductos()) {
            detalles.append(producto.toString());
        }

        return detalles.toString();
    }

    public String getEmpleadoActual() {
        return Session.getSession().getEmpleado().getNombre();
    }

    public void volverAVistaDeTrabajo() {
        Empleado empleado = uva.dis.negocio.modelos.Session.getSession().getEmpleado();
        switch (empleado.getRol()) {
            case 1: // Gerente
                GestorDeVistas.mostrarVistaTrabajoGerente();
                break;
            case 2: // Encargado
                GestorDeVistas.mostrarVistaTrabajoEncargado();
                break;
            case 3: // Vendedor
                GestorDeVistas.mostrarVistaTrabajoVendedor();
                break;
            default:
                GestorDeVistas.mostrarVistaIdentificarse();
                break;
        }
    }

    public void inicializarVista() {
        String nombre = getEmpleadoActual();
        vista.setNombreUsuario(nombre);
        String filtro = vista.getFiltroSeleccionado();
        obtenerPedidosSinRecoger(filtro);
    }

    public void verDetallesPedido() {
        PedidoDTO pedido = vista.getPedidoSeleccionado();
        if (pedido == null) {
            vista.mostrarError("Por favor, seleccione un pedido de la lista.");
            return;
        }

        String detalles = obtenerDetallesPedido(pedido);
        vista.mostrarDetalles(detalles);
        vista.mostrarError("");
    }

}
