package uva.dis.negocio.modelos;


public class Negocio {

    private TipoDeNegocio tipo;
    private String cif;
    private String nombre;
    private String denominacionOficial;
    private Direccion direccion;
    private Hora cierreVentaAlPublico;
    private Hora aperturaMejorNoTirarlo;
    private Fecha fechaInscripcion;
    private boolean verificado;
    private boolean superfiable;

    // Constructor privado que recibe builder
    private Negocio() {
    }

    public static class Builder {
        private final Negocio negocio;

        public Builder() {
            this.negocio = new Negocio();
        }

        public Builder tipo(TipoDeNegocio tipo) {
            negocio.tipo = tipo;
            return this;
        }

        public Builder cif(String cif) {
            negocio.cif = cif;
            return this;
        }

        public Builder nombre(String nombre) {
            negocio.nombre = nombre;
            return this;
        }

        public Builder denominacionOficial(String denominacionOficial) {
            negocio.denominacionOficial = denominacionOficial;
            return this;
        }

        public Builder direccion(Direccion direccion) {
            negocio.direccion = direccion;
            return this;
        }

        public Builder cierreVentaAlPublico(Hora cierre) {
            negocio.cierreVentaAlPublico = cierre;
            return this;
        }

        public Builder aperturaMejorNoTirarlo(Hora apertura) {
            negocio.aperturaMejorNoTirarlo = apertura;
            return this;
        }

        public Builder fechaInscripcion(Fecha fecha) {
            negocio.fechaInscripcion = fecha;
            return this;
        }

        public Builder verificado(boolean verificado) {
            negocio.verificado = verificado;
            return this;
        }

        public Builder superfiable(boolean superfiable) {
            negocio.superfiable = superfiable;
            return this;
        }

        public Negocio build() {
            return negocio;
        }
    }

    // Constructor auxiliar para solo el cif
    public Negocio(String cif) {
        this.cif = cif;
    }

    // Getters
    public TipoDeNegocio getTipo() {
        return tipo;
    }

    public String getCif() {
        return cif;
    }

    public String getNombre() {
        return nombre;
    }

    public String getDenominacionOficial() {
        return denominacionOficial;
    }

    public Direccion getDireccion() {
        return direccion;
    }

    public Hora getCierreVentaAlPublico() {
        return cierreVentaAlPublico;
    }

    public Hora getAperturaMejorNoTirarlo() {
        return aperturaMejorNoTirarlo;
    }

    public Fecha getFechaInscripcion() {
        return fechaInscripcion;
    }

    public boolean isVerificado() {
        return verificado;
    }

    public boolean isSuperfiable() {
        return superfiable;
    }
}
