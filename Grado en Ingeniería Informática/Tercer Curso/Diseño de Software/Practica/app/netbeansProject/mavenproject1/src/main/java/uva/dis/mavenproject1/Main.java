package uva.dis.mavenproject1;

import uva.dis.persistencia.DBConnection;
import java.util.logging.Level;
import java.util.logging.Logger;
import uva.dis.exception.ConfigurationFileNotFoundException;
import uva.dis.exception.ConfigurationReadException;
import uva.dis.exception.PersistenciaException;
import uva.dis.exception.ScriptExecutionException;
import uva.dis.interfaz.vistactrl.GestorDeVistas;

public class Main {


        public static void main(String[] args) throws PersistenciaException, ScriptExecutionException {
            try {
                DBConnection db = DBConnection.getInstance();
                db.openConnection();

                GestorDeVistas.mostrarVistaIdentificarse();

            } catch (ConfigurationFileNotFoundException | ConfigurationReadException | ClassNotFoundException ex) {
                Logger.getLogger(Main.class.getName()).log(Level.SEVERE, "Error en configuración o carga de driver", ex);
            }
        }

       
    }
