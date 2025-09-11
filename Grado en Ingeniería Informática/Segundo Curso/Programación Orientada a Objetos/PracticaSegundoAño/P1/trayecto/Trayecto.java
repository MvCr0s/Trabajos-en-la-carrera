package trayecto;

import java.time.LocalDate;
import es.uva.inf.poo.maps.GPSCoordinate;
import puerto.Puerto;
import muelle.Muelle;

/**
 * Representa un trayecto entre dos puertos a través de muelles y calcula su precio.
 */
public abstract class Trayecto {
    private Muelle muelleOrigen;
    private Puerto puertoOrigen;
    private LocalDate fechaInicio;
    private Muelle muelleDestino;
    private Puerto puertoDestino;
    private LocalDate fechaFin;
    private double precio;

    /**
     * Crea un trayecto entre dos puertos con sus muelles y fechas.
     * @param muelleOrigen Muelle de origen
     * @param puertoOrigen Puerto de origen
     * @param fechaInicio Fecha de inicio
     * @param muelleDestino Muelle de destino
     * @param puertoDestino Puerto de destino
     * @param fechaFin Fecha de fin
     */
    public Trayecto(Muelle muelleOrigen, Puerto puertoOrigen, LocalDate fechaInicio,
                    Muelle muelleDestino, Puerto puertoDestino, LocalDate fechaFin) {
        this.muelleOrigen = muelleOrigen;
        this.puertoOrigen = puertoOrigen;
        this.fechaInicio = fechaInicio;
        this.muelleDestino = muelleDestino;
        this.puertoDestino = puertoDestino;
        this.fechaFin = fechaFin;
    }

    public abstract double calcularPrecioTrayecto();

    public abstract String obtenerInformacionCompleta();

    public double getPrecio() throws NoSuchFieldException {
        if (precio != 0) {
            return precio;
        } else {
            throw new NoSuchFieldException("El precio no está establecido");
        }
    }

    public void setPrecio(double precio) {
        this.precio = precio;
    }

    public Muelle getMuelleOrigen() {
        return muelleOrigen;
    }

    public void setMuelleOrigen(Muelle muelleOrigen) {
        this.muelleOrigen = muelleOrigen;
    }

    public Puerto getPuertoOrigen() {
        return puertoOrigen;
    }

    public void setPuertoOrigen(Puerto puertoOrigen) {
        this.puertoOrigen = puertoOrigen;
    }

    public LocalDate getFechaInicio() {
        return fechaInicio;
    }

    public void setFechaInicio(LocalDate fechaInicio) {
        this.fechaInicio = fechaInicio;
    }

    public Muelle getMuelleDestino() {
        return muelleDestino;
    }

    public void setMuelleDestino(Muelle muelleDestino) {
        this.muelleDestino = muelleDestino;
    }

    public Puerto getPuertoDestino() {
        return puertoDestino;
    }

    public void setPuertoDestino(Puerto puertoDestino) {
        this.puertoDestino = puertoDestino;
    }

    public LocalDate getFechaFin() {
        return fechaFin;
    }

    public void setFechaFin(LocalDate fechaFin) {
        if (fechaFin.isAfter(fechaInicio)) {
            this.fechaFin = fechaFin;
        } else {
            throw new IllegalArgumentException("La fecha de llegada es anterior a la fecha de salida");
        }
    }

    public double obtenerDistanciaMillasMarinas() {
        GPSCoordinate coordenadasOrigen = muelleOrigen.getUbicacionGPS();
        GPSCoordinate coordenadasDestino = muelleDestino.getUbicacionGPS();

        double distanciaKilometros = coordenadasOrigen.getDistanceTo(coordenadasDestino);
        return distanciaKilometros / 1.60934;  // Convertir kilómetros a millas
    }
}
