/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.modelos;

import uva.dis.persistencia.TarjetaDeProductoDTO;
import uva.dis.exception.TipoEmpleadoIncorrectoException;

/**
 *
 * @author dediego
 */
public class Session {


    private static Session theSession;


    private Empleado empleadoIdentificado;
    private Negocio negocio;


    private TarjetaDeProductoDTO tarjetaTemporal;


    private Session() {
    }


    public static Session getSession() {
        if (theSession == null) {
            theSession = new Session();
        }
        return theSession;
    }


    public void setEmpleado(Empleado empleadoIdentificado) {
        this.empleadoIdentificado = empleadoIdentificado;
    }

   
    public Empleado getEmpleado() {
        return empleadoIdentificado;
    }

    public <T extends Empleado> T getEmpleadoAs(Class<T> tipo) throws TipoEmpleadoIncorrectoException {
        if (tipo.isInstance(empleadoIdentificado)) {
            return tipo.cast(empleadoIdentificado);
        }
        throw new TipoEmpleadoIncorrectoException("El empleado en la sesión no es instancia de " + tipo.getSimpleName());
    }

    public Negocio getNegocio() {
        return negocio;
    }

    public void setNegocio(Negocio negocio) {
        this.negocio = negocio;
    }

    // ----------- Tarjeta Temporal ----------------

    public TarjetaDeProductoDTO getTarjetaTemporal() {
        return tarjetaTemporal;
    }

    public void setTarjetaTemporal(TarjetaDeProductoDTO tarjetaTemporal) {
        this.tarjetaTemporal = tarjetaTemporal;
    }

    public void limpiarTarjetaTemporal() {
        this.tarjetaTemporal = null;
    }

    public static void close() {
    theSession = null;
    }

}