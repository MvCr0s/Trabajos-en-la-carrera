package uva.dis.negocio.controladorescasouso;

import java.io.IOException;
import uva.dis.persistencia.PedidoDAO;
import uva.dis.negocio.modelos.EstadoPedido;

import java.io.StringReader;
import java.time.LocalDateTime;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonReader;
import uva.dis.persistencia.PedidoDTO;
import uva.dis.persistencia.ProductoDTO;
import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.exception.PersistenciaException;
import uva.dis.negocio.modelos.Session;

public class ControladorCUConsultarPedidos {

    private static final Logger logger = Logger.getLogger(ControladorCUConsultarPedidos.class.getName());

    private ControladorCUConsultarPedidos() {
        // Constructor privado para evitar instanciación
    }

    public static List<PedidoDTO> buscarPedidosSinRecoger(String idNegocio, String filtro) {
        List<PedidoDTO> pedidos = new ArrayList<>();

        try {
            String pedidosJson = PedidoDAO.getPedidosSinRecogerPorNegocio(idNegocio, filtro);
            try (JsonReader reader = Json.createReader(new StringReader(pedidosJson))) {
                JsonArray pedidosArray = reader.readArray();
                for (JsonObject pedidoJson : pedidosArray.getValuesAs(JsonObject.class)) {
                    procesarPedidoJson(pedidoJson, pedidos);
                }
            }
        } catch (IOException | ClassNotFoundException | ConfigurationFileNotFoundException
                | ConfigurationReadException | PersistenciaException e) {
            logger.log(Level.SEVERE, "Error general al obtener o procesar pedidos: {0}", e.getMessage());
        }
        return pedidos;
    }

    private static void procesarPedidoJson(JsonObject pedidoJson, List<PedidoDTO> pedidos) throws PersistenciaException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        try {
            PedidoDTO pedido = construirPedidoDesdeJson(pedidoJson);
            pedidos.add(pedido);
        } catch (DateTimeParseException dtpe) {
            logger.log(Level.WARNING, "Error parseando fecha del pedido {0}: {1}",
                    new Object[]{pedidoJson.getString("id", "desconocido"), dtpe.getMessage()});
        } catch (IllegalArgumentException | NullPointerException e) {
            logger.log(Level.SEVERE, "Error construyendo el pedido: {0}", e.getMessage());
        }
    }

    
    private static PedidoDTO construirPedidoDesdeJson(JsonObject pedidoJson) throws PersistenciaException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.S");

        String id = pedidoJson.getString("id");
        String fechaHoraStr = pedidoJson.getString("fechaYHora");
        int estadoInt = pedidoJson.getInt("estado");

        LocalDateTime fechaHora = LocalDateTime.parse(fechaHoraStr, formatter);
        LocalDate fecha = fechaHora.toLocalDate();
        LocalTime hora = fechaHora.toLocalTime();
        EstadoPedido estado = EstadoPedido.values()[estadoInt - 1];

        PedidoDTO pedido = new PedidoDTO(id, fecha, hora, estado);
        List<ProductoDTO> productos = PedidoDAO.getProductosDePedido(id);
        pedido.setProductos(productos);

        return pedido;
    }

    public static String getNegocio() {
        return Session.getSession().getEmpleado().getIdNegocio();
    }
}
