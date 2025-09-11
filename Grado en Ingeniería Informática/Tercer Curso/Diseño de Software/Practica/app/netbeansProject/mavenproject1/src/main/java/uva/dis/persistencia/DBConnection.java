package uva.dis.persistencia;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;

import java.util.Properties;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.exception.PersistenciaException;
import uva.dis.exception.ScriptExecutionException;

public class DBConnection {

     private static DBConnection theDBConnection;
    private Connection conn = null;

    private String url;
    private String user;
    private String password;

    private DBConnection(String url, String user, String password) throws ClassNotFoundException {
        this.url = url;
        this.user = user;
        this.password = password;
        Class.forName("org.apache.derby.jdbc.ClientDriver");
    }

    public static DBConnection getInstance() throws ConfigurationFileNotFoundException, ConfigurationReadException, ClassNotFoundException {
        if (theDBConnection == null) {
            Properties prop = new Properties();
            try (InputStream read = DBConnection.class.getResourceAsStream("/uva/dis/mavenproject1/config.db")) {
                if (read == null) throw new ConfigurationFileNotFoundException("DB configuration file not found.");
                prop.load(read);
                String url = prop.getProperty("url");
                String user = prop.getProperty("user");
                String password = prop.getProperty("password");

                theDBConnection = new DBConnection(url, user, password);
            } catch (IOException e) {
                throw new ConfigurationReadException("Couldn't read DB configuration file.", e);
            }
        }
        return theDBConnection;
    }

     public void openConnection() throws PersistenciaException, ScriptExecutionException {
        try {
            conn = DriverManager.getConnection(url, user, password);
            runScript("/uva/dis/mavenproject1/createTables.sql");
            runScript("/uva/dis/mavenproject1/populate.sql");
        } catch (SQLException ex) {
            throw new PersistenciaException("Error al abrir la conexión con la base de datos", ex);
        }
    }

    public void closeConnection() throws PersistenciaException {
        try {
            if (conn != null && !conn.isClosed()) {
                conn.close();
            }
        } catch (SQLException ex) {
            throw new PersistenciaException("Error al cerrar la conexión", ex);
        }
    }

    public PreparedStatement getStatement(String s) throws PersistenciaException {
        try {
            return conn.prepareStatement(s);
        } catch (SQLException ex) {
            throw new PersistenciaException("Error en la obtención del statement", ex);
        }
    }

    private void runScript(String scriptFileName) throws ScriptExecutionException {
        InputStream inputStream = getClass().getResourceAsStream(scriptFileName);
        if (inputStream == null) {
            throw new ScriptExecutionException("Script file not found: " + scriptFileName, null);
        }

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream)); Statement stmt = conn.createStatement()) {

            StringBuilder sqlBuilder = new StringBuilder();
            String line;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("--")) {
                    continue;
                }
                sqlBuilder.append(line).append(" ");
                if (line.endsWith(";")) {
                    String sql = sqlBuilder.toString().trim();
                    sql = sql.substring(0, sql.length() - 1);
                    stmt.execute(sql);
                    sqlBuilder.setLength(0);
                }
            }
        } catch (IOException | SQLException e) {
            throw new ScriptExecutionException("Error ejecutando el script: " + scriptFileName, e);
        }
    }

    public Connection getConnection() {
        return conn;
    }

}
