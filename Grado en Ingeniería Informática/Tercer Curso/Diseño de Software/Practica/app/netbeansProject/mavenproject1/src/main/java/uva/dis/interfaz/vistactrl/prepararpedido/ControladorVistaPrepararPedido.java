/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.interfaz.vistactrl.prepararpedido;

import java.util.List;
import uva.dis.persistencia.PedidoDTO;
import uva.dis.persistencia.ProductoDTO;
import uva.dis.exception.PrepararPedidoException;
import uva.dis.negocio.modelos.Empleado;
import uva.dis.negocio.modelos.Session;
import uva.dis.interfaz.vistactrl.GestorDeVistas;
import uva.dis.negocio.controladorescasouso.ControladorCUPrepararPedido;
/**
 *
 * @author Admin
 */
public class ControladorVistaPrepararPedido {

    private final ControladorCUPrepararPedido controladorCU;
    private final PrepararPedidoVista vista;

    public ControladorVistaPrepararPedido(PrepararPedidoVista vista) {
        this.vista = vista;
        controladorCU = new ControladorCUPrepararPedido();
    }

    public void inicializarVista() {
        vista.mostrarError("");
        vista.mostrarUsuario(getEmpleadoActual());
        cargarPedidos(false);
    }

    public List<PedidoDTO> obtenerPedidosRealizadosHoy() {
        return controladorCU.buscarPedidosRealizadosHoy();
    }

    public String obtenerDetallesPedido(PedidoDTO pedido) {
        StringBuilder detalles = new StringBuilder(pedido.toString2());
        try {
            List<ProductoDTO> productos = controladorCU.obtenerProductosDePedido(pedido.getId());
            pedido.setProductos(productos);
            for (ProductoDTO producto : productos) {
                detalles.append(producto.toString());
            }
        } catch (Exception e) {
            detalles.append("\nError al cargar productos del pedido: ").append(e.getMessage());
        }
        return detalles.toString();
    }

    
    public void marcarPedidoComoPreparado(String idPedido) throws PrepararPedidoException {
    controladorCU.marcarPedidoComoPreparado(idPedido);
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

    public void seleccionarPedido() {
        PedidoDTO pedido = vista.getPedidoSeleccionado();
        if (pedido != null) {
            vista.setSeleccionado(true);
            String detalles = obtenerDetallesPedido(pedido);
            vista.mostrarDetalles(detalles);
            vista.mostrarError("");
        } else {
            vista.mostrarError("Por favor, seleccione un pedido de la lista.");
        }
    }

    public void prepararPedido() {
        PedidoDTO pedidoSeleccionado = vista.getPedidoSeleccionado();
        vista.mostrarError("");
        if (pedidoSeleccionado != null && vista.getSeleccionado()) {
            int confirmacion = javax.swing.JOptionPane.showConfirmDialog(
                    null,
                    "¿Confirmas preparar este pedido?",
                    "Confirmar preparación",
                    javax.swing.JOptionPane.YES_NO_OPTION);

            if (confirmacion == javax.swing.JOptionPane.YES_OPTION) {
                try {
                    marcarPedidoComoPreparado(pedidoSeleccionado.getId());
                    vista.mostrarError("Pedido marcado como preparado correctamente.");
                    cargarPedidos(true);
                } catch (Exception e) {
                    vista.mostrarError("Error al actualizar el estado del pedido: " + e.getMessage());
                }
            } else {
                vista.mostrarError("Preparación cancelada.");
            }
        } else {
            vista.mostrarError("Por favor, seleccione un pedido de la lista.");
        }
        vista.setSeleccionado(false);
    }

    public void cargarPedidos(boolean fuePreparado) {
        List<PedidoDTO> listaPedidos = obtenerPedidosRealizadosHoy();

        vista.mostrarDetalles("");

        if (listaPedidos.isEmpty()) {
            vista.mostrarLista(new javax.swing.DefaultListModel<>());
            if (fuePreparado) {
                vista.mostrarError("Último pedido preparado. No quedan más pedidos sin recoger.");
            } else {
                vista.mostrarError("No se ha encontrado ningún pedido sin recoger actualmente.");
            }
        } else {
            javax.swing.DefaultListModel<PedidoDTO> model = new javax.swing.DefaultListModel<>();
            for (PedidoDTO pedido : listaPedidos) {
                model.addElement(pedido);
            }
            vista.mostrarLista(model);
        }
    }

}
