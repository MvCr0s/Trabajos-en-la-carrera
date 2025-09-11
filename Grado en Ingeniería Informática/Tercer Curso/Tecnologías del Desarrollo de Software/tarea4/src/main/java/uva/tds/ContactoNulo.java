package uva.tds;

public class ContactoNulo extends Contacto {
    public ContactoNulo() {
        super("N/A", "N/A");
    }

    @Override
    public String getNombre() {
        return "Desconocido";
    }

    @Override
    public String getApellido() {
        return "Desconocido";
    }
}
