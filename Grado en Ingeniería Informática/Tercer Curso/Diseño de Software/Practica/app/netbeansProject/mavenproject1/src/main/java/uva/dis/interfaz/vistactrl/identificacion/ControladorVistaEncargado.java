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
public class ControladorVistaEncargado {

    private final TrabajoEncargadoVista vista;

    public ControladorVistaEncargado(TrabajoEncargadoVista vista) {
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
            case "Preparar pedido":
                GestorDeVistas.mostrarVistaPrepararPedido();
                break;
            case "Añadir producto":
                GestorDeVistas.mostrarVistaBuscarTarjeta();
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
        String[] tareas = {
            "Consultar datos de empleado",
            "Consultar pedidos sin recoger",
            "Entregar pedido",
            "Preparar pedido",
            "Añadir producto",
            "Eliminar producto",         
        };
        
        vista.setListaTareas(tareas);
    }

    public void realizarTareaSeleccionada() {
        String tareaSeleccionada = vista.getTareaSeleccionada();
        realizarTarea(tareaSeleccionada);
    }

}
