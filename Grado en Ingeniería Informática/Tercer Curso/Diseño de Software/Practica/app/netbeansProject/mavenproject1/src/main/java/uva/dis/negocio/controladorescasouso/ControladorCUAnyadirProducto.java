/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.controladorescasouso;

import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.persistencia.ProductoDAO;
import uva.dis.persistencia.TarjetaDeProductoDTO;
import uva.dis.exception.CrearProductoException;
import uva.dis.exception.PersistenciaException;
/**
 *
 * @author dediego
 */
public class ControladorCUAnyadirProducto {

    public void crearProducto(int idTarjeta, double medida, int cantidad, double precio) throws CrearProductoException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        try {
            ProductoDAO.insertarProducto(idTarjeta, medida, cantidad, precio);
        } catch (PersistenciaException e) {
            throw new CrearProductoException("Error al crear el producto", e);
        }
    }

    public TarjetaDeProductoDTO getTarjetaActual() {
        return uva.dis.negocio.modelos.Session.getSession().getTarjetaTemporal();
    }
}
