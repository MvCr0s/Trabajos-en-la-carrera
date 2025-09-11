import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import org.antlr.v4.gui.TreeViewer;
import javax.swing.*;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Analizador {
    public static void main(String[] args) throws Exception {

        InputStream inputStream = new FileInputStream(args[0]);
        CharStream input = CharStreams.fromStream(inputStream);
        SimpleClassModelWithConstraintsLexer lexer = new SimpleClassModelWithConstraintsLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        SimpleClassModelWithConstraintsParser parser = new SimpleClassModelWithConstraintsParser(tokens);
        ParseTree tree = parser.model();
        ParseTreeWalker walker = new ParseTreeWalker();
        AnalizadorSemanticoListener listener = new AnalizadorSemanticoListener();

        walker.walk(listener, tree);

        if (listener.hasErrors()) {
            Map<String, List<String>> atributosPorClase = listener.getAtributosPorClase();
            Map<String, List<String>> invariantesPorClase = listener.getInvariantesPorClase();
            Map<String, List<GeneradorCodigoJava.Asociacion>> asociacionesPorClase = listener.getAsociacionesPorClase();
            Map<String, List<GeneradorCodigoJava.Operacion>> operacionesPorClase = listener.getOperacionesPorClase();
            Set<String> clasesAsociacion = listener.getClasesAsociacion();
            GeneradorCodigoJava.generarCodigo(
                    listener.getClassNames(),
                    atributosPorClase,
                    invariantesPorClase,
                    asociacionesPorClase,
                    operacionesPorClase,
                    listener.getEnumerationNames(),
                    listener.getCamposPorDatatype(),
                    clasesAsociacion,
                    listener.getLiteralesPorEnum());

            System.out.println("Código Java generado en ModeloGenerado.java");
        }

        if (listener.hasErrors()) {
            System.out.println("\n -------------------------- ERRORES DETECTADOS ---------------------");
            for (String err : listener.getSemanticErrors()) {
                System.err.println(err);
            }
            System.exit(1);
        }

        System.out.println("Errores no detectados");
    }

}
