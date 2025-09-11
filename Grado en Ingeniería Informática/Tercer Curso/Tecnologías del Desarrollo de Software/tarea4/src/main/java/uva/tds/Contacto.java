package uva.tds;

public class Contacto {
    private String nombre;
    private String apellido;

    public Contacto(String nombre, String apellido) {
        if (nombre == null || nombre.trim().isEmpty()){ 
            throw new IllegalArgumentException("El nombre no puede ser nulo o vacío");
        }

        this.nombre = nombre;
        setApellido(apellido);
    }

    public String getNombre() {
        return nombre;
    }

    public String getApellido() {
        return apellido;
    }

    public void setApellido(String apellido) {
        if (apellido == null || apellido.trim().isEmpty()) {
            throw new IllegalArgumentException("El apellido no puede ser nulo o vacío");
        }
        this.apellido = apellido;
    }
}
