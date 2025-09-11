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
public class ControladorVistaVendedor {

    private final TrabajoVendedorVista vista;

    public ControladorVistaVendedor(TrabajoVendedorVista vista) {
        this.vista = vista;
    }

    public void realizarTarea(String tareaSeleccionada) {
        if (tareaSeleccionada == null) {
            vista.mostrarMensajeError("Por favor, selecciona una tarea a realizar.");
            return;
        }

        if(tareaSeleccionada.equalsIgnoreCase("Consultar pedidos sin recoger")) {
            GestorDeVistas.mostrarVistaConsultarPedidos();
        } else{
            vista.mostrarMensajeError("Opción aún no implementada.");
                
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
            "Consultar pedidos sin recoger"       
         
        });
    }

    public void realizarTareaSeleccionada() {
        String tareaSeleccionada = vista.getTareaSeleccionada();
        realizarTarea(tareaSeleccionada);
    }

}
