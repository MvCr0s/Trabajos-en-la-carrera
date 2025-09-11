package uva.dis.negocio.controladorescasouso;

import java.io.StringReader;
import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonReader;
import uva.dis.persistencia.TarjetaDeProductoDAO;
import uva.dis.persistencia.TarjetaDeProductoDTO;
import uva.dis.negocio.modelos.Session;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
/**
 *
 * @author dediego
 */
public class ControladorCUBuscarTarjeta {

    public TarjetaDeProductoDTO verificarNombreTarjeta(String nombre) throws Exception {
        String negocio = Session.getSession().getNegocio().getCif();
        
        try{
            
            String tarjetaJsonString = TarjetaDeProductoDAO.buscarPorNombreYNegocio(nombre, negocio);
            
            if(tarjetaJsonString.equals("{}")){
                return null;
            }
            
            JsonReader reader = Json.createReader(new StringReader(tarjetaJsonString));
            JsonObject tarjetaJSON = reader.readObject();

            int id = tarjetaJSON.getInt("id");
            String nombre2 = tarjetaJSON.getString("nombre");
            Short unidad = (short) tarjetaJSON.getInt("unidad");
            String descripcion = tarjetaJSON.getString("descripcion");
            String alergenos = tarjetaJSON.getString("alergenos");
            String ingredientes = tarjetaJSON.getString("ingredientes");
            String negocio2 = tarjetaJSON.getString("negocio");           
           
            return new TarjetaDeProductoDTO(id, nombre2,unidad,descripcion,alergenos,ingredientes,negocio2);
            
        } catch(Exception ex){
            return null;
        }
    }

}
