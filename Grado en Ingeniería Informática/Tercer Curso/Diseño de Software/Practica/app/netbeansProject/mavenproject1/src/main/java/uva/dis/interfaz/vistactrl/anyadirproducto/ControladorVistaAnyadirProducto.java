/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.interfaz.vistactrl.anyadirproducto;

import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.persistencia.TarjetaDeProductoDTO;
import uva.dis.exception.CrearProductoException;
import uva.dis.interfaz.vistactrl.GestorDeVistas;
import uva.dis.negocio.controladorescasouso.ControladorCUAnyadirProducto;

/**
 *
 * @author dediego
 */
public class ControladorVistaAnyadirProducto {

    private final AnyadirProductoVista vista;
    private final ControladorCUAnyadirProducto controladorCU;

    public ControladorVistaAnyadirProducto(AnyadirProductoVista vista) {
        this.vista = vista;
        this.controladorCU = new ControladorCUAnyadirProducto();
    }

    public void crearProducto(double medida, int cantidad, double precio) throws CrearProductoException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        TarjetaDeProductoDTO tarjeta = controladorCU.getTarjetaActual();
        if (tarjeta == null) {
            throw new IllegalStateException("No se ha proporcionado una tarjeta descriptora.");
        }
        controladorCU.crearProducto(tarjeta.getId(), medida, cantidad, precio);
    }

    public void procesarFormulario(String precioStr, String medidaStr, String cantidadStr) throws ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        TarjetaDeProductoDTO tarjeta = controladorCU.getTarjetaActual();
        try {
            // Limpieza básica para decimales y símbolos
            String precioLimpio = precioStr.replace("€", "").replace(",", ".").trim();
            String medidaLimpia = medidaStr.replace(",", ".").trim();
            String cantidadLimpia = cantidadStr.trim();

            double precio = Double.parseDouble(precioLimpio);
            double medida = Double.parseDouble(medidaLimpia);
            int cantidad = Integer.parseInt(cantidadLimpia);

            if (cantidad < 1) {
                vista.mostrarError("La cantidad debe ser un número entero mayor o igual que 1.");
                return;
            }
            if (precio < 0) {
                vista.mostrarError("El precio debe ser mayor o igual que 0.");
                return;
            }
            if (medida < 0) {
                vista.mostrarError("La medida debe ser mayor o igual que 0.");
                return;
            }

            controladorCU.crearProducto(tarjeta.getId(), medida, cantidad, precio);

            vista.mostrarError("Se han añadido " + cantidad + " productos correctamente.");
            vista.deshabilitarBotonContinuar();

        } catch (NumberFormatException e) {
            vista.mostrarError("Error: introduce valores numéricos válidos.");
        } catch (CrearProductoException e) {
            vista.mostrarError("Error interno: " + e.getMessage());
        }
    }

    public void volverAVistaDeTrabajo() {
        uva.dis.negocio.modelos.Session.getSession().limpiarTarjetaTemporal();
        GestorDeVistas.mostrarVistaBuscarTarjeta();
    }

    public TarjetaDeProductoDTO getTarjeta() {
        return controladorCU.getTarjetaActual();
    }

    public void mostrarTarjetaDescriptoraEnVista() {
        TarjetaDeProductoDTO tarjeta = controladorCU.getTarjetaActual();

        if (tarjeta == null) {
            return;
        }

        String unidadTxt;
        switch (tarjeta.getUnidad()) {
            case 1:
                unidadTxt = "kg";
                break;
            case 2:
                unidadTxt = "litros";
                break;
            default:
                unidadTxt = "unidades";
                break;
        }
        String detalles = "Unidad: " + unidadTxt + "\n"
                + "Descripción: " + tarjeta.getDescripcion() + "\n"
                + "Ingredientes: " + tarjeta.getIngredientes() + "\n"
                + "Alérgenos: " + tarjeta.getAlergenos();

        vista.mostrarTarjeta(tarjeta.getNombre(), detalles);
    }

}
