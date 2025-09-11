/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.controladorescasouso;

import java.io.StringReader;
import uva.dis.persistencia.TarjetaDeProductoDAO;
import uva.dis.negocio.modelos.Session;
import javax.json.Json;
import javax.json.JsonException;
import javax.json.JsonObject;
import javax.json.JsonReader;
import uva.dis.persistencia.TarjetaDeProductoDTO;
import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.exception.CrearTarjetaException;
import uva.dis.exception.PersistenciaException;
import uva.dis.exception.VerificarTarjetaException;

/**
 *
 * @author dediego
 */
public class ControladorCUAnyadirTarjetaDescriptora {

    public TarjetaDeProductoDTO verificarNombreTarjeta(String nombre) throws VerificarTarjetaException  {
        String negocio = Session.getSession().getNegocio().getCif();

        try {
            String tarjetaJsonString = TarjetaDeProductoDAO.buscarPorNombreYNegocio(nombre, negocio);
            TarjetaDeProductoDTO tarjetaVista = null;

            if (!tarjetaJsonString.equals("{}")) {
                JsonReader reader = Json.createReader(new StringReader(tarjetaJsonString));
                JsonObject tarjetaJSON = reader.readObject();

                int id = tarjetaJSON.getInt("id");
                String nombre2 = tarjetaJSON.getString("nombre");
                Short unidad = (short) tarjetaJSON.getInt("unidad");
                String descripcion = tarjetaJSON.getString("descripcion");
                String alergenos = tarjetaJSON.getString("alergenos");
                String ingredientes = tarjetaJSON.getString("ingredientes");
                String negocio2 = tarjetaJSON.getString("negocio");
                tarjetaVista = new TarjetaDeProductoDTO(id, nombre2, unidad, descripcion, alergenos, ingredientes, negocio2);
            }

            return tarjetaVista;

        } catch (Exception ex) {
            throw new VerificarTarjetaException ("Error al verificar nombre de tarjeta", ex);
        }
    }

    public TarjetaDeProductoDTO crearTarjeta(String nombre, String descripcion, String alergenos, String ingredientes, Short unidad) throws Exception {
        try {
            String negocio = Session.getSession().getNegocio().getCif();
            int nuevoId = uva.dis.persistencia.TarjetaDeProductoDAO.obtenerSiguienteId(); 

            TarjetaDeProductoDAO.insertarTarjeta(nombre, unidad, descripcion, alergenos, ingredientes, negocio);

 
            return new TarjetaDeProductoDTO(nuevoId, nombre, unidad, descripcion, alergenos, ingredientes, negocio);

        } catch (PersistenciaException | JsonException e) {       
            throw new PersistenciaException("Error de persistencia al insertar la tarjeta: " + e.getMessage(), e);
        }
    }

    public TarjetaDeProductoDTO crearTarjeta2(String nombre, String descripcion, String alergenos, String ingredientes, String unidad) throws CrearTarjetaException {
        short unidadNum = convertirUnidad(unidad);
        String negocio = Session.getSession().getNegocio().getCif();

        int nuevoId;
        try {
            nuevoId = TarjetaDeProductoDAO.obtenerSiguienteId();
        } catch (PersistenciaException | ConfigurationFileNotFoundException
                | ConfigurationReadException | ClassNotFoundException ex) {
            throw new CrearTarjetaException("Error al obtener el siguiente ID para la tarjeta", ex);
        }
        return new TarjetaDeProductoDTO(nuevoId, nombre, unidadNum, descripcion, alergenos, ingredientes, negocio);
    }

    private short convertirUnidad(String unidadTexto) {
        String unidad = unidadTexto.toLowerCase();
        if (unidad.equalsIgnoreCase("kilogramos")) {
            return 1;
        }
        if (unidad.equalsIgnoreCase("litros")) {
            return 2;
        }
        if (unidad.equalsIgnoreCase("unidades")) {
            return 3;
        }
        return -1;
    }

}
