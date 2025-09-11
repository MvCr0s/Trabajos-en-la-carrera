/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.interfaz.vistactrl.identificacion;

import uva.dis.negocio.modelos.Session;
import uva.dis.interfaz.vistactrl.GestorDeVistas;

/**
 *
 * @author dediego
 */
public class ControladorVistaGerente {

    private final TrabajoGerenteVista vista;

    public ControladorVistaGerente(TrabajoGerenteVista vista) {
        this.vista = vista;
    }

    public void realizarTarea(String tareaSeleccionada) {
        if (tareaSeleccionada == null) {
            vista.mostrarMensajeError("Por favor, selecciona una tarea a realizar.");
            return;
        }

        switch (tareaSeleccionada) {
            case "Consultar pedidos sin recoger":
                GestorDeVistas.mostrarVistaConsultarPedidos();
                break;
            case "Añadir tarjeta descriptora de producto":
                GestorDeVistas.mostrarVistaAnyadirTarjeta();
                break;
            case "Añadir producto":
                GestorDeVistas.mostrarVistaBuscarTarjeta();
                break;
            case "Preparar pedido":
                GestorDeVistas.mostrarVistaPrepararPedido();
                break;
            default:
                vista.mostrarMensajeError("Opción aún no implementada.");
                break;
        }
    }

    public void salir() {
        Session.close();
        GestorDeVistas.mostrarVistaIdentificarse();
        vista.dispose();
    }

    public void inicializarVista() {
        vista.setNombreUsuario(uva.dis.negocio.modelos.Session.getSession().getEmpleado().getNombre());
        vista.setListaTareas(new String[]{
            "Consultar datos de empleado",
            "Entregar pedido",
            "Consultar pedidos sin recoger",
            "Preparar pedido",
            "Eliminar producto",
            "Añadir producto",
            "Añadir tarjeta descriptora de producto",
            "Consultar quejas de clientes",
            "Añadir empleados"
        });
    }

    public void realizarTareaSeleccionada() {
        String tarea = vista.getTareaSeleccionada();
        realizarTarea(tarea);
    }

}
