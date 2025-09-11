/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.interfaz.vistactrl.anyadirtarjetadescriptora;

import uva.dis.persistencia.TarjetaDeProductoDTO;
import uva.dis.exception.CrearTarjetaException;
import uva.dis.exception.VerificarTarjetaException;
import uva.dis.negocio.modelos.Empleado;
import uva.dis.negocio.modelos.Session;
import uva.dis.interfaz.vistactrl.GestorDeVistas;
import uva.dis.interfaz.vistactrl.anyadirproducto.AnyadirProductoVista;
import uva.dis.negocio.controladorescasouso.ControladorCUAnyadirTarjetaDescriptora;

/**
 *
 * @author Admin
 */
public class ControladorVistaAnyadirTarjetaDescriptora {

    private final AnyadirTarjetaDescriptoraVista vista;
    private final ControladorCUAnyadirTarjetaDescriptora cuController;

    public ControladorVistaAnyadirTarjetaDescriptora(AnyadirTarjetaDescriptoraVista vista) {
        this.vista = vista;
        this.cuController = new ControladorCUAnyadirTarjetaDescriptora();
    }

    public String getEmpleadoActual() {
        return uva.dis.negocio.modelos.Session.getSession().getEmpleado().getNombre();
    }

    public TarjetaDeProductoDTO verificarNombreTarjeta(String nombre) throws VerificarTarjetaException {
        return cuController.verificarNombreTarjeta(nombre);
    }

    public TarjetaDeProductoDTO crearTarjeta1(String nombre, String descripcion, String alergenos, String ingredientes, Short unidad) throws Exception {
        return cuController.crearTarjeta(nombre, descripcion, alergenos, ingredientes, unidad);

    }

    public TarjetaDeProductoDTO crearTarjeta2(String nombre, String descripcion, String alergenos, String ingredientes, String unidad) throws CrearTarjetaException {
        return cuController.crearTarjeta2(nombre, descripcion, alergenos, ingredientes, unidad);
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

    public String formatearDetalles(String nombre, String unidad, String descripcion, String alergenos, String ingredientes) {
        return "Nombre: " + nombre + "\n"
                + "Unidad: " + unidad + "\n"
                + "Descripción: " + descripcion + "\n"
                + "Alérgenos: " + alergenos + "\n"
                + "Ingredientes: " + ingredientes;
    }

    public String formatearDetalles(TarjetaDeProductoDTO dto) {
        String unidadTxt;
        switch (dto.getUnidad()) {
            case 1:
                unidadTxt = "kilogramos";
                break;
            case 2:
                unidadTxt = "litros";
                break;
            case 3:
                unidadTxt = "unidades";
                break;
            default:
                unidadTxt = "¿?";
                break;
        }
        return "Nombre: " + dto.getNombre() + "\n"
                + "Unidad: " + unidadTxt + "\n"
                + "Descripción: " + dto.getDescripcion() + "\n"
                + "Ingredientes: " + dto.getIngredientes() + "\n"
                + "Alérgenos: " + dto.getAlergenos();
    }

    public void buscarTarjeta() {
        String nombre = vista.getNombreProducto();
        if (nombre.isEmpty()) {
            vista.mostrarError("Introduce un nombre válido");
            return;
        }
        try {
            TarjetaDeProductoDTO dto = verificarNombreTarjeta(nombre);
            if (dto != null) {
                vista.setTarjetaActual(dto);
                vista.mostrarError("Se ha encontrado una tarjeta existente para el nombre indicado.");
                vista.mostrarBotonVerDetalles();
            } else {
                vista.mostrarError("No se ha encontrado una tarjeta. Puede crear una nueva.");
                vista.mostrarBotonCrearTarjeta();
            }
        } catch (Exception ex) {
            vista.mostrarError("Error al buscar: " + ex.getMessage());
        }
    }

    public void procesarCrearTarjeta() {
        if (vista.getTextoCrearTarjeta().equalsIgnoreCase("Ver detalles")) {
            vista.mostrarDetallesExistente();
        } else {
            vista.prepararFormularioNuevaTarjeta();
        }
    }

    public void previsualizarTarjeta() {
        String nombre = vista.getNombreProducto().trim();
        String descripcion = vista.getDescripcion().trim();
        String alergenos = vista.getAlergenos().trim();
        String ingredientes = vista.getIngredientes().trim();
        String unidad = vista.getUnidadSeleccionada();

        if (nombre.isEmpty() || descripcion.isEmpty() || alergenos.isEmpty() || ingredientes.isEmpty() || unidad == null) {
            vista.mostrarError("Rellena todos los campos");
            return;
        }

        try {
            TarjetaDeProductoDTO dto = cuController.crearTarjeta2(nombre, descripcion, alergenos, ingredientes, unidad);
            vista.setTarjetaActual(dto);
            vista.mostrarDetalles(formatearDetalles(nombre, unidad, descripcion, alergenos, ingredientes));
            vista.cambiarACard("card4");
            vista.mostrarError("");
        } catch (Exception e) {
            vista.mostrarError("Error al crear tarjeta: " + e.getMessage());
        }
    }

    public void confirmarCreacionTarjeta() {
        TarjetaDeProductoDTO dto = vista.getTarjetaActual();
        try {
            cuController.crearTarjeta(dto.getNombre(), dto.getDescripcion(), dto.getAlergenos(), dto.getIngredientes(), dto.getUnidad());
            vista.mostrarError("Tarjeta creada correctamente.");
            vista.ocultarBotonesConfirmacion();
            vista.mostrarBotonCrearProducto();
        } catch (Exception e) {
            vista.mostrarError("Error al crear tarjeta: " + e.getMessage());
        }
    }

    public void cancelarCreacionTarjeta() {
        String nombre = vista.getNombreProducto();
        vista.mostrarLabelYCard("Producto: " + nombre, "card3");
        vista.mostrarError("");
    }

    public void iniciarCreacionProducto() {
        Session.getSession().setTarjetaTemporal(vista.getTarjetaActual());
        AnyadirProductoVista vistaProducto = new AnyadirProductoVista();
        vistaProducto.setVisible(true);
        vista.setVisible(false);
    }

}
