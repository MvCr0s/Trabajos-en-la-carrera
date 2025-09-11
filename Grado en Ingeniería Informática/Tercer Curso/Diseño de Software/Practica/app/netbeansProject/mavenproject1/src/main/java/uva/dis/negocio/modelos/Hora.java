/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.modelos;

/**
 *
 * @author Admin
 */
public class Hora {
    
    private int vhora;
    private int minutos;
    

    public Hora(int hora, int minutos) {
        this.vhora = hora;
        this.minutos = minutos;
    }

    public int getHora() {
        return vhora;
    }

    public void setHora(int hora) {
        this.vhora = hora;
    }

    public int getMinutos() {
        return minutos;
    }

    public void setMinutos(int minutos) {
        this.minutos = minutos;
    }
    
   
    
    
}
