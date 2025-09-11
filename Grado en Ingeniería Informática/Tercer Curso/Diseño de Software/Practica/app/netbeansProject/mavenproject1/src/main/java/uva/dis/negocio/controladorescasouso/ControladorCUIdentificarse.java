package uva.dis.negocio.controladorescasouso;

import uva.dis.negocio.modelos.Empleado;
import uva.dis.persistencia.EmpleadoDAO;
import uva.dis.persistencia.EmpleadoDTO;
import uva.dis.negocio.modelos.Session;
import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonReader;
import java.io.StringReader;
import java.util.logging.Level;
import java.util.logging.Logger;
import uva.dis.negocio.modelos.Negocio;

public class ControladorCUIdentificarse {

    private static final Logger logger = Logger.getLogger(ControladorCUIdentificarse.class.getName());

    private ControladorCUIdentificarse() {
        // Constructor vacío intencionalmente: la inicialización se realiza manualmente más adelante.
    }

    public static EmpleadoDTO comprobarIdentificacion(String nif2, String contrasenaIngresada) {

        try {
            String empleadoJsonString = EmpleadoDAO.getEmpleadoConNifYPassword(nif2, contrasenaIngresada);
            if (empleadoJsonString.equals("{}")) {
                return null;
            }

            JsonReader reader = Json.createReader(new StringReader(empleadoJsonString));
            JsonObject empleadoJSON = reader.readObject();

            String nif = empleadoJSON.getString("nif");
            String nombre = empleadoJSON.getString("nombre");
            String contrasena = empleadoJSON.getString("password");
            String email = empleadoJSON.getString("email");
            String idNegocio = empleadoJSON.getString("negocio");
            int rol = empleadoJSON.getInt("rol");
            int vinculo = empleadoJSON.getInt("vinculo");
            int disponibilidad = empleadoJSON.getInt("disponibilidad");

            Empleado empleadoIdentificado = new Empleado(nif, nombre, email, contrasena, idNegocio, rol);
            Session.getSession().setEmpleado(empleadoIdentificado);
            Session.getSession().setNegocio(new Negocio(idNegocio));

            boolean estaVinculadoNormal = (vinculo == 1);
            boolean estaDisponible = (disponibilidad == 3);
            boolean activo = estaVinculadoNormal && estaDisponible;

            return new EmpleadoDTO(empleadoIdentificado.getNombre(), activo, true, rol);

        } catch (Exception ex) {
            logger.log(Level.SEVERE, "Error en identificación: {0}", ex.getMessage());
            return null;
        }
    }
}
