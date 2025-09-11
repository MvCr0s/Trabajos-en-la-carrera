/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.persistencia;

import java.io.StringWriter;
import java.sql.*;
import java.util.Optional;
import java.sql.Connection;
import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonObjectBuilder;
import javax.json.JsonWriter;
import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.exception.JsonConversionException;
import uva.dis.exception.PersistenciaException;

/**
 *
 * @author dediego
 */
public class TarjetaDeProductoDAO {
    
    private TarjetaDeProductoDAO() {
         // Constructor vacío intencionalmente: la inicialización se realiza manualmente más adelante.
    }

    public static String buscarPorNombreYNegocio(String nombre, String negocio) throws Exception {
        Connection conn = DBConnection.getInstance().getConnection();
        String tarjetaJsonString = "";

        String query = "SELECT Id, Nombre, Unidad, Descripcion, Alergenos, Ingredientes, Negocio FROM TARJETASDEPRODUCTOS WHERE Nombre = ? AND Negocio = ?";

        try (PreparedStatement stmt = conn.prepareStatement(query)) {
            stmt.setString(1, nombre);
            stmt.setString(2, negocio);
            ResultSet rs = stmt.executeQuery();

            if (rs.next()) {
                JsonObject tarjetaJson = obtainTarjetaJson(
                        rs.getInt("Id"),
                        rs.getString("Nombre"),
                        rs.getShort("Unidad"),
                        rs.getString("Descripcion"),
                        rs.getString("Alergenos"),
                        rs.getString("Ingredientes"),
                        rs.getString("Negocio")
                );
                
                try(StringWriter stringWriter = new StringWriter(); JsonWriter writer = Json.createWriter(stringWriter)){
                    writer.writeObject(tarjetaJson);
                    tarjetaJsonString = stringWriter.toString();
                }
            } else {
                tarjetaJsonString = "{}";
            }
            rs.close();
            return tarjetaJsonString; 
           
        } catch(SQLException ex){
            throw new PersistenciaException("Error en la obtención de los datos", ex);
        }

    }

    public static void insertarTarjeta(String nombre, Short unidadNum, String descripcion, String alergenos, String ingredientes, String negocio) throws PersistenciaException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException{
        Connection conn = DBConnection.getInstance().getConnection();
        int id = obtenerSiguienteId();

        String sql = "INSERT INTO TARJETASDEPRODUCTOS (Id, Nombre, Unidad, Descripcion, Alergenos, Ingredientes, Negocio) VALUES (?, ?, ?, ?, ?, ?, ?)";
        try (PreparedStatement stmt = conn.prepareStatement(sql)) {
            stmt.setInt(1, id);
            stmt.setString(2, nombre);
            stmt.setShort(3, unidadNum);
            stmt.setString(4, descripcion);
            stmt.setString(5, alergenos);
            stmt.setString(6, ingredientes);
            stmt.setString(7, negocio);
            stmt.executeUpdate();
        }  catch (SQLException e) {
            throw new PersistenciaException("Error en la insercion de datos", e);        
        }
    }

    public static int obtenerSiguienteId() throws PersistenciaException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        Connection conn = DBConnection.getInstance().getConnection();
        String query = "SELECT MAX(Id) FROM TARJETASDEPRODUCTOS";
        try (PreparedStatement stmt = conn.prepareStatement(query)) {
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                return rs.getInt(1) + 1;
            } else {
                return 1;
            }
        } catch (SQLException ex) {
            throw new PersistenciaException("Error en la obtención de los datos", ex);
        }
    }

    private static JsonObject obtainTarjetaJson(int id, String nombre, short unidad, String descripcion, String alergenos, String ingredientes, String negocio) throws JsonConversionException {
        try {
            JsonObjectBuilder builder = Json.createObjectBuilder()
                .add("id", id)
                .add("nombre", nombre)
                .add("unidad", unidad)
                .add("descripcion", descripcion)
                .add("alergenos", alergenos)
                .add("ingredientes", ingredientes)
                .add("negocio", negocio);
            
            return builder.build();
            
        } catch (Exception ex) {
            throw new JsonConversionException("Error en la conversión de los datos a JSON", ex);
        }     
    }
}
