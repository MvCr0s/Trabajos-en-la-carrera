/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.controladorescasouso;

import java.util.List;
import uva.dis.persistencia.PedidoDAO;
import uva.dis.persistencia.PedidoDTO;
import uva.dis.persistencia.ProductoDTO;
import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.exception.ConsultaProductosPedidoException;
import uva.dis.exception.PersistenciaException;
import uva.dis.exception.PrepararPedidoException;
import uva.dis.negocio.modelos.Session;

/**
 *
 * @author dediego
 */
public class ControladorCUPrepararPedido {

    public List<PedidoDTO> buscarPedidosRealizadosHoy() {
        String idNegocio = Session.getSession().getEmpleado().getIdNegocio();
        return ControladorCUConsultarPedidos.buscarPedidosSinRecoger(idNegocio, "Realizados");
    }

    public List<ProductoDTO> obtenerProductosDePedido(String idPedido) throws ConsultaProductosPedidoException {
        try {
            return PedidoDAO.getProductosDePedido(idPedido);
        } catch (PersistenciaException
                | ConfigurationFileNotFoundException
                | ConfigurationReadException
                | ClassNotFoundException e) {
            throw new ConsultaProductosPedidoException("Error al obtener los productos del pedido", e);
        }
    }

    public void marcarPedidoComoPreparado(String idPedido) throws PrepararPedidoException {
        try {
            PedidoDAO.marcarPedidoComoPreparado(idPedido);
        } catch (PersistenciaException | ConfigurationFileNotFoundException
                | ConfigurationReadException | ClassNotFoundException e) {
            throw new PrepararPedidoException("No se pudo marcar el pedido como preparado", e);
        }
    }

}
