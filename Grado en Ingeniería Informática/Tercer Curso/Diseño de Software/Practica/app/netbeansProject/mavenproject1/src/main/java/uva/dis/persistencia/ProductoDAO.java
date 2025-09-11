/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.persistencia;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.time.LocalDate;
import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.exception.PersistenciaException;

public class ProductoDAO {
    
    private ProductoDAO() {
         // Constructor vacío intencionalmente: la inicialización se realiza manualmente más adelante.a
    }
    private static long ultimoTimestamp = 0;
    private static int contador = 0;

    public static void insertarProducto(int idTarjeta, double medida, int cantidad, double precio) throws PersistenciaException, ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        
        Connection conn = DBConnection.getInstance().getConnection();
        String sql = "INSERT INTO PRODUCTOS (ID, PRECIO, FECHA, CANTIDADDISPONIBLE, MEDIDA, DESCRIPCION) VALUES (?, ?, ?, ?, ?, ?)";

        try (PreparedStatement stmt = conn.prepareStatement(sql)) {
            stmt.setString(1, generarNuevoID());
            stmt.setDouble(2, precio);
            stmt.setDate(3, java.sql.Date.valueOf(LocalDate.now()));
            stmt.setInt(4, cantidad);
            stmt.setDouble(5, medida);
            stmt.setInt(6, idTarjeta);
            stmt.executeUpdate();
        } catch (SQLException e) {
            throw new PersistenciaException("Error en la inserción de los datos", e);
        }
    }

    private static synchronized String generarNuevoID() {
        long actual = System.currentTimeMillis() % 1000000000000L; 
        if (actual == ultimoTimestamp) {
            contador++;
        } else {
            contador = 0;
            ultimoTimestamp = actual;
        }
        return "P" + actual + contador; 
    }
}

