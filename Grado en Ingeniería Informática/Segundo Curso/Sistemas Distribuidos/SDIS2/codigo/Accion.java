import java.io.Serializable;

public class Accion implements Serializable {
  private static final long serialVersionUID = 1L;
  public String op;
  public String arg;

  public Accion(String op, String arg) {
    this.op = op;
    this.arg = arg;
  }

  @Override
  public String toString() {
    return "<" + op + ":" + arg + ">";
  }
}
