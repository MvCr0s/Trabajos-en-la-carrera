/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.interfaz.vistactrl.identificacion;

import uva.dis.persistencia.EmpleadoDTO;
import uva.dis.interfaz.vistactrl.GestorDeVistas;
import uva.dis.negocio.controladorescasouso.ControladorCUIdentificarse;


/**
 *
 * @author dediego
 */
public class ControladorVistaIdentificarse {
    
    private final IdentificarseVista vista;
   
    public ControladorVistaIdentificarse( IdentificarseVista vista) {    
        this.vista = vista;
    }

    
public void continuarIdentificacion(String nif, String contrasena) {
    
    EmpleadoDTO empleadoVista = ControladorCUIdentificarse.comprobarIdentificacion(nif, contrasena);

    if (empleadoVista == null) {
        vista.mostrarMensajeError("Credenciales incorrectas.");      
    }  
    else if (empleadoVista.isActivo()) {
        cerrarVista();  
       
        switch (empleadoVista.getRol()) {
            case 1:
                GestorDeVistas.mostrarVistaTrabajoGerente();
                break;
            case 2:
                GestorDeVistas.mostrarVistaTrabajoEncargado();
                break;
            default:
                GestorDeVistas.mostrarVistaTrabajoVendedor();
                break;
        }
                       
    } else {
        vista.mostrarMensajeError("El usuario no se encuentra activo en este momento.");  
    }
    
}
    
    public void cerrarVista() {
        if (vista != null) {
            vista.dispose();  // Cierra la ventana
        }
    }
    
}

