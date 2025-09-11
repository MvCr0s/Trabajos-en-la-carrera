package uva.tds;

import java.time.LocalDate;


/**
 * Hello world!
 */
public class Factura {
    private String asunto;
    private LocalDate fecha;
    private double importe;


    public Factura(String asunto, LocalDate fecha, double importe){
        if(asunto==null || asunto.trim().isEmpty()){throw new IllegalArgumentException();}
        if(importe<0){throw new IllegalArgumentException();}
        this.asunto=asunto;
        this.fecha=fecha;
        this.importe=importe;
    }


    public String getAsunto(){
        return asunto;
    }

    public LocalDate getFecha(){
        return fecha;
    }

    public double getImporte(){
        return importe;
    }


    /*@Override
    public boolean equals(Object obj){
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Factura factura = (Factura) obj;
        if(asunto != factura.getAsunto()){return false;}
        if(importe != factura.getImporte()){return false;}
        if(!fecha.equals(factura.getFecha())){return false;}
        return true;

    }*/

    @Override
    public boolean equals(Object obj){
        if(this==obj){return true;}
        if(obj==null || getClass()!=obj.getClass()){return false;}
        Factura f = (Factura) obj;
        if(asunto!=f.getAsunto()){return false;}
        if(importe != f.getImporte()){return false;}
        if(!fecha.equals(f.getFecha())){return false;}
        return true;
    }
}
