/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.persistencia;

/**
 *
 * @author Admin
 */
public class EmpleadoDTO {
    private final String nombre;
    private final boolean activo;
    private final boolean encontrado;
    private final int rol;
    

    public EmpleadoDTO(String nombre, boolean activo, boolean encontrado, int rol) {
        this.nombre = nombre;
        this.activo = activo;
        this.encontrado = encontrado;
        this.rol = rol;
    }

    public String getNombre() {
        return nombre;
    }

    public boolean isActivo() {
        return activo;
    }

    public boolean isEncontrado() {
        return encontrado;
    }
    
    public int getRol(){
        return rol;
    }
    
    
    
}
