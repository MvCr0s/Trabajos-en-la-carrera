

/**
 * No se puede crear el personaje
 *
 * @author CesarLlamas
 * @since  04-15-2024
 */

public class BadPersonajeException extends Exception {
  /* msg: lista de personajes disponibles */
  public BadPersonajeException(String msg) {
    super(msg);
  }
}
