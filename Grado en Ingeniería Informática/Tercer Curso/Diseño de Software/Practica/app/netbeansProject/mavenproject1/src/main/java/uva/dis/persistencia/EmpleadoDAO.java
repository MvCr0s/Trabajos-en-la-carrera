package uva.dis.persistencia;

import java.io.StringWriter;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import javax.json.Json;
import javax.json.JsonObjectBuilder;
import javax.json.JsonObject;
import javax.json.JsonWriter;
import uva.dis.exception.JsonConversionException;
import uva.dis.exception.PersistenciaException;
import uva.dis.negocio.modelos.Empleado;

/**
 *
 * @author dediego
 */
public class EmpleadoDAO {

    private EmpleadoDAO() {
        // Constructor vacío intencionalmente: la inicialización se realiza manualmente más adelante.
    }

    public static String getEmpleadoConNifYPassword(String nif, String password) throws Exception {
        DBConnection db = DBConnection.getInstance();
        String empleadoJsonString = "";

        String query = "SELECT e.*, r.Rol, v.Vinculo, d.Disponibilidad "
                + "FROM EMPLEADOSENEGOCIOSSUSCRITOS e "
                + "JOIN ROLESENEMPRESA r ON e.Nif = r.Empleado "
                + "JOIN TIPOSDEROL t ON r.Rol = t.IdTipo "
                + "LEFT JOIN VINCULACIONESCONLAEMPRESA v ON e.Nif = v.Empleado "
                + "LEFT JOIN DISPONIBILIDADES d ON e.Nif = d.Empleado "
                + "WHERE e.Nif = ? AND e.Password = ?";

        try (PreparedStatement stmt = db.getStatement(query)) {
            stmt.setString(1, nif);
            stmt.setString(2, password);
            ResultSet result = stmt.executeQuery();

            if (result.next()) {

                Empleado empleado = new Empleado(
                        result.getString("Nif"),
                        result.getString("Nombre"),
                        result.getString("Email"),
                        result.getString("Password"),
                        result.getString("Negocio"),
                        result.getInt("Rol")
                );

               
                JsonObject empleadoJson = obtainEmpleadoJson(
                        empleado,
                        result.getInt("Vinculo"),
                        result.getInt("Disponibilidad")
                );

                try (StringWriter stringWriter = new StringWriter(); JsonWriter writer = Json.createWriter(stringWriter)) {
                    writer.writeObject(empleadoJson);
                    empleadoJsonString = stringWriter.toString();
                }
            } else {
                empleadoJsonString = "{}";
            }

            result.close();
            return empleadoJsonString;
        } catch (SQLException ex) {
            throw new PersistenciaException("Error en la obtención de los datos", ex);
        }
    }

    private static JsonObject obtainEmpleadoJson(Empleado empleado, int vinculo, int disponibilidad) throws Exception {

        try {
            JsonObjectBuilder builder = Json.createObjectBuilder()
                    .add("nif", empleado.getNif())
                    .add("nombre", empleado.getNombre())
                    .add("password", empleado.getContrasena())
                    .add("email", empleado.getEmail())
                    .add("negocio", empleado.getIdNegocio())
                    .add("rol", empleado.getRol())
                    .add("vinculo", vinculo)
                    .add("disponibilidad", disponibilidad);

            return builder.build();
        } catch (Exception ex) {
            throw new JsonConversionException("Error en la conversión de los datos a JSON", ex);
        }
    }
}
