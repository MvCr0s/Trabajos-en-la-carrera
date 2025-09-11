/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.interfaz.vistactrl;

/**
 *
 * @author Admin
 */
import javax.swing.JFrame;
import uva.dis.interfaz.vistactrl.identificacion.IdentificarseVista;
import uva.dis.interfaz.vistactrl.identificacion.TrabajoEncargadoVista;
import uva.dis.interfaz.vistactrl.identificacion.TrabajoGerenteVista;
import uva.dis.interfaz.vistactrl.identificacion.TrabajoVendedorVista;

public class GestorDeVistas {

    private static JFrame vistaActual;
    
    private GestorDeVistas(){
        
    }

    private static void mostrarVista(JFrame nuevaVista) {
        if (vistaActual != null) {
            vistaActual.setVisible(false);
            vistaActual.dispose();
        }
        vistaActual = nuevaVista;
        vistaActual.setVisible(true);
    }

    public static void mostrarVistaIdentificarse() {
        mostrarVista(new IdentificarseVista());
    }

    public static void mostrarVistaTrabajoGerente() {
        mostrarVista(new TrabajoGerenteVista());
    }

    public static void mostrarVistaTrabajoEncargado( ) {
        mostrarVista(new TrabajoEncargadoVista());
    }

    public static void mostrarVistaTrabajoVendedor( ) {
        mostrarVista(new TrabajoVendedorVista());
    }

    public static void mostrarVistaConsultarPedidos( ) {
        mostrarVista(new uva.dis.interfaz.vistactrl.consultarpedido.ConsultarPedidoVista());
    }

    public static void mostrarVistaPrepararPedido( ) {
        mostrarVista(new uva.dis.interfaz.vistactrl.prepararpedido.PrepararPedidoVista());
    }

    public static void mostrarVistaAnyadirTarjeta( ) {
        mostrarVista(new uva.dis.interfaz.vistactrl.anyadirtarjetadescriptora.AnyadirTarjetaDescriptoraVista());
    }

    public static void mostrarVistaAnyadirProducto( ) {
        mostrarVista(new uva.dis.interfaz.vistactrl.anyadirproducto.AnyadirProductoVista());
    }

    public static void mostrarVistaBuscarTarjeta( ) {
        mostrarVista(new uva.dis.interfaz.vistactrl.anyadirproducto.BuscarTarjetaDescriptoraVista());
    }

}
