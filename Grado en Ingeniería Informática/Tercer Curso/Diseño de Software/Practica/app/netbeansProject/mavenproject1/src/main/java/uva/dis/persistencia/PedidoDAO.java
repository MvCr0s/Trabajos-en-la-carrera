/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.persistencia;

import java.io.IOException;
import java.io.StringWriter;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import javax.json.Json;
import javax.json.JsonArrayBuilder;
import javax.json.JsonObjectBuilder;
import javax.json.JsonWriter;
import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.exception.PersistenciaException;
import uva.dis.negocio.modelos.Producto;

/**
 *
 * @author dediego
 */
public class PedidoDAO {
    
    private PedidoDAO() {
         // Constructor vacío intencionalmente: la inicialización se realiza manualmente más adelante.a
    }

    public static String getPedidosSinRecogerPorNegocio(String idNegocio, String filtro) throws PersistenciaException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException, IOException {
        DBConnection db = DBConnection.getInstance();
        String pedidosJsonString = "";

        String seleccionPedidos;
        switch (filtro) {
            case "Realizados":
                seleccionPedidos = "(1)";
                break;
            case "Preparados":
                seleccionPedidos = "(2)";
                break;
            case "Todos":
            default:
                seleccionPedidos = "(1,2)";
                break;
        }

        String query = "SELECT Id, FechaYHora, Estado FROM PEDIDOS " +
                       "WHERE Negocio = ? AND Estado IN " + seleccionPedidos + " " +
                       "AND DATE(FechaYHora) = CURRENT_DATE";

        try (PreparedStatement stmt = db.getStatement(query)) {
            stmt.setString(1, idNegocio);

            try (ResultSet result = stmt.executeQuery()) {
                JsonArrayBuilder pedidosArray = Json.createArrayBuilder();
                while (result.next()) {
                    JsonObjectBuilder pedidoBuilder = Json.createObjectBuilder()
                            .add("id", result.getString("Id"))
                            .add("fechaYHora", result.getTimestamp("FechaYHora").toString())
                            .add("estado", result.getInt("Estado"));
                    pedidosArray.add(pedidoBuilder);
                }
                try (StringWriter stringWriter = new StringWriter(); JsonWriter writer = Json.createWriter(stringWriter)) {
                    writer.writeArray(pedidosArray.build());
                    pedidosJsonString = stringWriter.toString();
                }
            }
            return pedidosJsonString;
        } catch (SQLException ex) {
            throw new PersistenciaException("Error en la obtención de los datos", ex);
        }

}
    
public static List<ProductoDTO> getProductosDePedido(String idPedido) throws PersistenciaException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {

    List<ProductoDTO> productosDTO = new ArrayList<>();
    Connection conn = DBConnection.getInstance().getConnection();


    String query =
        "SELECT tp.Nombre, p.Medida, um.NombreUnidad, lp.CantidadEnPedido, tp.Descripcion, p.Precio " +
        "FROM LINEASDEPEDIDOS lp " +
        "JOIN PRODUCTOS p ON lp.Producto = p.Id " +
        "JOIN TARJETASDEPRODUCTOS tp ON p.Descripcion = tp.Id " +
        "JOIN UNIDADESDEMEDIDA um ON tp.Unidad = um.IdUnidad " +
        "WHERE lp.Pedido = ?";

    try (PreparedStatement stmt = conn.prepareStatement(query)) {
        stmt.setString(1, idPedido);
        ResultSet rs = stmt.executeQuery();

        while (rs.next()) {
            String nombre = rs.getString("Nombre");
            double medida = rs.getDouble("Medida");
            String unidad = rs.getString("NombreUnidad");
            String descripcion = rs.getString("Descripcion");
            double precio = rs.getDouble("Precio");
            int cantidad = rs.getInt("CantidadEnPedido");

            Producto producto = new Producto(0, nombre, medida, unidad, descripcion, precio); // ID no se extrae, por eso 0
            productosDTO.add(producto.toDTO(cantidad));
        }

        rs.close();
        return productosDTO;
        } catch (SQLException ex) {
        throw new PersistenciaException("Error en la obtención de los datos", ex);
    }
}




    public static void marcarPedidoComoPreparado(String idPedido) throws PersistenciaException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        DBConnection db = DBConnection.getInstance();
        String query = "UPDATE PEDIDOS SET Estado = 2 WHERE Id = ?";

        try (PreparedStatement stmt = db.getStatement(query)) {
            stmt.setString(1, idPedido);
            stmt.executeUpdate();
        } catch (SQLException e) {
            throw new PersistenciaException("Error en la actualización de los datos", e);
        }
}



}