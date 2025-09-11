/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package uva.dis.negocio.modelos;

/**
 *
 * @author Admin
 */
public class Direccion {
    
    private String nombreVia;
    private int numero;
    private String otros;
    private String localidad;
    private String municipio;
    private String provincia;
    private String codigoPostal;

    public Direccion(String nombreVia, int numero, String otros, String localidad, String municipio, String provincia, String codigoPostal) {
        this.nombreVia = nombreVia;
        this.numero = numero;
        this.otros = otros;
        this.localidad = localidad;
        this.municipio = municipio;
        this.provincia = provincia;
        this.codigoPostal = codigoPostal;
    }

    public String getNombreVia() {
        return nombreVia;
    }

    public void setNombreVia(String nombreVia) {
        this.nombreVia = nombreVia;
    }

    public int getNumero() {
        return numero;
    }

    public void setNumero(int numero) {
        this.numero = numero;
    }

    public String getOtros() {
        return otros;
    }

    public void setOtros(String otros) {
        this.otros = otros;
    }

    public String getLocalidad() {
        return localidad;
    }

    public void setLocalidad(String localidad) {
        this.localidad = localidad;
    }

    public String getMunicipio() {
        return municipio;
    }

    public void setMunicipio(String municipio) {
        this.municipio = municipio;
    }

    public String getProvincia() {
        return provincia;
    }

    public void setProvincia(String provincia) {
        this.provincia = provincia;
    }

    public String getCodigoPostal() {
        return codigoPostal;
    }

    public void setCodigoPostal(String codigoPostal) {
        this.codigoPostal = codigoPostal;
    }
    
}
