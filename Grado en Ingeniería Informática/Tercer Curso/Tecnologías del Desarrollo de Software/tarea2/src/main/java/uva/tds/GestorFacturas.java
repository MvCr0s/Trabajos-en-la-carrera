package uva.tds;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class GestorFacturas {
    private LocalDate fechaInicio;
    private LocalDate fechaFin;
    private String nombre;
    private boolean estado;
    private ArrayList<Factura> facturas;

    public GestorFacturas(LocalDate fechaInicio, LocalDate fechaFin, String nombre){
        if(fechaFin == null || fechaInicio==null || nombre==null || nombre.trim().isEmpty() || nombre.trim().length()>10){throw new IllegalArgumentException();}
        if(fechaFin.isBefore(fechaInicio)){throw new IllegalArgumentException();}

        this.estado=true;
        this.fechaFin=fechaFin;
        this.fechaInicio=fechaInicio;
        this.nombre=nombre;
        this.facturas= new ArrayList<>();
    }

    // Getters
    public LocalDate getFechaInicio() {
        return fechaInicio;
    }

    public LocalDate getFechaFin() {
        return fechaFin;
    }

    public String getNombre() {
        return nombre;
    }

    public boolean isEstado() {
        return estado;
    }

    public ArrayList<Factura> getFacturas(){
        return facturas;
    }


    public void setEstado(boolean estado) {
        this.estado = estado;
    }

    public void añadir(Factura f){
        if(f==null){throw new IllegalArgumentException();}
        if(f.getFecha().isBefore(fechaInicio) || !isEstado()){throw new IllegalStateException();}
        for(int i=0; i < facturas.size(); i++){
            if(f.equals(facturas.get(i))){throw new IllegalStateException();}
        }
        facturas.add(f);
    }

    public void añadirFacturas(ArrayList<Factura> f){
        if(f==null){throw new IllegalArgumentException();}
        for(int i = 0; i<f.size(); i++){
            for(int j=i+1; j<f.size(); j++){
                if(f.get(i).equals(f.get(j))){throw new IllegalStateException();}
            }
        }

        for(int i = 0; i<f.size(); i++){
            añadir(f.get(i));
        }

    }

    public List<Factura> listarFacturasPorFecha() {
        return facturas.stream().sorted(Comparator.comparing(Factura::getFecha)).toList();
    }


    public List<Factura> listarFacturasPorImporte(){
        return facturas.stream().sorted(Comparator.comparing(Factura::getImporte)).toList();
    }
}
