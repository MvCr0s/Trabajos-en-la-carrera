/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.interfaz.vistactrl.anyadirproducto;

import uva.dis.persistencia.TarjetaDeProductoDTO;
import uva.dis.negocio.modelos.Empleado;
import uva.dis.negocio.modelos.Session;
import uva.dis.interfaz.vistactrl.GestorDeVistas;
import uva.dis.negocio.controladorescasouso.ControladorCUBuscarTarjeta;
/**
 *
 * @author dediego
 */
public class ControladorVistaBuscarTarjeta {

    private final BuscarTarjetaDescriptoraVista vista;
    private final ControladorCUBuscarTarjeta controladorCU;

    public ControladorVistaBuscarTarjeta(BuscarTarjetaDescriptoraVista vista) {
        this.vista = vista;
        this.controladorCU = new ControladorCUBuscarTarjeta();
    }

    public void buscarTarjeta(String nombre) {
        try {
            TarjetaDeProductoDTO dto = controladorCU.verificarNombreTarjeta(nombre);
            if (dto != null) {

                vista.setTarjetaActual(dto);
                vista.mostrarMensaje("Tarjeta existente encontrada. Puede crear producto.");
                vista.mostrarBotonCrear("Crear producto");
            } else {
                
                vista.mostrarMensaje("No se ha encontrado ninguna tarjeta.");
                vista.setTarjetaActual(null);
            }
        } catch (Exception e) {
            vista.mostrarMensaje("Error al buscar: " + e.getMessage());
        }
    }

    public void procesarCrearProducto() {
        TarjetaDeProductoDTO tarjeta = vista.getTarjetaActual();
        if (tarjeta != null) {
            Session.getSession().setTarjetaTemporal(tarjeta);
            GestorDeVistas.mostrarVistaAnyadirProducto();
        } else {
            vista.mostrarMensaje("No hay una tarjeta válida seleccionada.");
        }
    }

    public String getEmpleadoActual() {
        return Session.getSession().getEmpleado().getNombre();
    }

    public void volverAVistaDeTrabajo() {
        Empleado empleado = Session.getSession().getEmpleado();
        switch (empleado.getRol()) {
            case 1:
                GestorDeVistas.mostrarVistaTrabajoGerente();
                break;
            case 2:
                GestorDeVistas.mostrarVistaTrabajoEncargado();
                break;
            case 3:
                GestorDeVistas.mostrarVistaTrabajoVendedor();
                break;
            default:
                GestorDeVistas.mostrarVistaIdentificarse();
                break;
        }
    }
}
