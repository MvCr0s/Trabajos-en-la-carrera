package uva.dis.negocio.modelos;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

/**
 *
 * @author Admin
 */
public class Empleado {
    private final String nif;
    private final String nombre;
    private final String contrasena;
    private final String email;
    private final String idNegocio;
    private final int rol;
    
    public Empleado(String nif, String nombre, String email ,String contrasena, String idNegocio, int rol) {
        this.nif = nif;
        this.nombre = nombre;
        this.contrasena = contrasena;
        this.email = email;
        this.idNegocio = idNegocio;
        this.rol = rol;
    }

    public String getIdNegocio() {
        return idNegocio;
    }
    
    public String getNif() {
        return nif;
    }

    public String getNombre() {
        return nombre;
    }

    public String getContrasena() {
        return contrasena;
    }

    public String getEmail() {
        return email;
    }
    
    public int getRol() {
        return rol;
    }   
    
  
}
