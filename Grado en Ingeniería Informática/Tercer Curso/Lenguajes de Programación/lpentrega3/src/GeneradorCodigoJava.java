import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class GeneradorCodigoJava {

    private static final Map<String, String> TYPE_MAPPING = Map.of(
            "Integer", "int",
            "Real", "float",
            "Boolean", "boolean",
            "String", "String");

    private static final String OUTPUT_DIR = "salidas/";

    public static void generarCodigo(Set<String> classNames,
            Map<String, List<String>> atributosPorClase,
            Map<String, List<String>> invariantesPorClase,
            Map<String, List<Asociacion>> asociacionesPorClase,
            Map<String, List<Operacion>> operacionesPorClase,
            Set<String> enumerationNames,
            Map<String, List<String>> camposPorDatatype,
            Set<String> clasesAsociacion,
            Map<String, List<String>> literalesPorEnumeracion) throws IOException {

        // Crea el directorio si no existe
        new java.io.File(OUTPUT_DIR).mkdirs();

        // Procesar clases asociación con estrategia de reificación
        procesarClasesAsociacion(clasesAsociacion, classNames, atributosPorClase,
                asociacionesPorClase, invariantesPorClase);

        // === 1. Clases normales ===
        try (FileWriter writer = new FileWriter(OUTPUT_DIR + "ModeloGenerado.java")) {

            // Imports necesarios para colecciones
            writer.write("import java.util.*;\n\n");

            for (String clase : classNames) {
                writer.write("class " + clase + " {\n\n");

                List<String> atributos = atributosPorClase.getOrDefault(clase, List.of());
                for (String linea : atributos) {
                    String[] partes = linea.split(":");
                    if (partes.length == 2) {
                        String nombre = partes[0].trim();
                        String tipo = partes[1].trim();
                        tipo = TYPE_MAPPING.getOrDefault(tipo, tipo);
                        writer.write("    private " + tipo + " " + nombre + ";\n");
                    }
                }

                // Atributos de asociaciones
                List<Asociacion> asociaciones = asociacionesPorClase != null
                        ? asociacionesPorClase.getOrDefault(clase, List.of())
                        : List.of();
                for (Asociacion asoc : asociaciones) {
                    if (asoc != null) {
                        String tipoAtributo = obtenerTipoAtributoAsociacion(asoc);
                        writer.write("    private " + tipoAtributo + " " + asoc.nombreAtributo + ";\n");
                    }
                }

                writer.write("\n    public " + clase + "() {\n");
                writer.write("        // TODO\n");

                // Inicializar colecciones de asociaciones
                for (Asociacion asoc : asociaciones) {
                    if (asoc != null && requiereInicializacion(asoc)) {
                        String implementacion = obtenerImplementacionColeccion(asoc);
                        writer.write("        " + asoc.nombreAtributo + " = new " + implementacion + "();\n");
                    }
                }

                List<String> invariantes = invariantesPorClase.getOrDefault(clase, List.of());
                for (int i = 0; i < invariantes.size(); i++) {
                    String expr = invariantes.get(i);
                    String nombreMetodo = expr.startsWith("@")
                            ? "check" + expr.substring(1, expr.indexOf(':'))
                            : "check" + (i + 1);
                    writer.write("        assert(" + nombreMetodo + "());\n");
                }

                writer.write("    }\n\n");

                for (int i = 0; i < invariantes.size(); i++) {
                    String expr = invariantes.get(i);
                    String nombreMetodo, comentario;
                    if (expr.startsWith("@")) {
                        int sep = expr.indexOf(':');
                        nombreMetodo = "check" + expr.substring(1, sep);
                        comentario = expr.substring(sep + 1);
                    } else {
                        nombreMetodo = "check" + (i + 1);
                        comentario = expr;
                    }

                    writer.write("    public boolean " + nombreMetodo + "() {\n");
                    writer.write("        // TODO " + comentario + "\n");
                    writer.write("        return false;\n");
                    writer.write("    }\n\n");
                }

                // Métodos para operaciones
                List<Operacion> operaciones = operacionesPorClase != null
                        ? operacionesPorClase.getOrDefault(clase, List.of())
                        : List.of();
                for (Operacion op : operaciones) {
                    if (op != null) {
                        generarMetodoOperacion(writer, op);
                    }
                }

                // === Getters y Setters ===
                for (String linea : atributos) {
                    String[] partes = linea.split(":");
                    if (partes.length == 2) {
                        String nombre = partes[0].trim();
                        String tipo = TYPE_MAPPING.getOrDefault(partes[1].trim(), partes[1].trim());

                        String nombreMayus = Character.toUpperCase(nombre.charAt(0)) + nombre.substring(1);

                        writer.write("    public " + tipo + " get" + nombreMayus + "() {\n");
                        writer.write("        return " + nombre + ";\n");
                        writer.write("    }\n\n");

                        writer.write("    public void set" + nombreMayus + "(" + tipo + " " + nombre + ") {\n");
                        writer.write("        this." + nombre + " = " + nombre + ";\n");
                        writer.write("    }\n\n");
                    }
                }

                writer.write("}\n\n");
            }

            // === 2. Enumeraciones ===
            for (String enumName : enumerationNames) {
                writer.write("public enum " + enumName + " {\n");

                List<String> literales = literalesPorEnumeracion.getOrDefault(enumName, List.of());
                for (int i = 0; i < literales.size(); i++) {
                    String literal = literales.get(i).toUpperCase();
                    writer.write("    " + literal);
                    if (i != literales.size() - 1) {
                        writer.write(",");
                    }
                    writer.write("\n");
                }

                writer.write("}\n\n");
            }

            // === 3. Datatypes como records Java 17 ===
            for (Map.Entry<String, List<String>> entry : camposPorDatatype.entrySet()) {
                String dtName = entry.getKey();
                List<String> campos = entry.getValue();

                writer.write("record " + dtName + "(");
                List<String> parametros = new ArrayList<>();
                for (String campo : campos) {
                    String[] partes = campo.split(":");
                    if (partes.length == 2) {
                        String nombre = partes[0].trim();
                        String tipo = TYPE_MAPPING.getOrDefault(partes[1].trim(), partes[1].trim());
                        parametros.add(tipo + " " + nombre);
                    }
                }
                writer.write(String.join(", ", parametros));
                writer.write(") {}\n\n");
            }
        }
    }

    // Estrategia de reificación para clases asociación
    private static void procesarClasesAsociacion(Set<String> clasesAsociacion,
            Set<String> classNames,
            Map<String, List<String>> atributosPorClase,
            Map<String, List<Asociacion>> asociacionesPorClase,
            Map<String, List<String>> invariantesPorClase) {

        for (String claseAsoc : clasesAsociacion) {
            // 1. Añadir la clase asociación como clase normal
            classNames.add(claseAsoc);

            // 2. Buscar las asociaciones relacionadas con esta clase asociación
            // y transformar la asociación original en referencias a la clase asociación
            reificarAsociaciones(claseAsoc, asociacionesPorClase);

            System.out.println("INFO: Clase asociación '" + claseAsoc +
                    "' reificada como clase normal con referencias bidireccionales.");
        }
    }

    private static void reificarAsociaciones(String claseAsoc,
            Map<String, List<Asociacion>> asociacionesPorClase) {
        // Buscar asociaciones que involucren esta clase asociación
        // y modificar las clases extremas para referenciar la clase asociación

        for (Map.Entry<String, List<Asociacion>> entry : asociacionesPorClase.entrySet()) {
            String clase = entry.getKey();
            List<Asociacion> asociaciones = entry.getValue();

            // Buscar asociaciones que terminen en la clase asociación
            for (int i = 0; i < asociaciones.size(); i++) {
                Asociacion asoc = asociaciones.get(i);

                if (asoc != null && asoc.tipoDestino.equals(claseAsoc)) {
                    // Esta es una asociación hacia una clase asociación
                    // Modificar para que sea una colección de la clase asociación

                    asociaciones.set(i, new Asociacion(asoc.nombreAtributo, claseAsoc, "*", false, false));
                }
            }
        }
        // Añadir referencias inversas desde la clase asociación hacia las clases
        // extremas
        // Esto se haría analizando el modelo original, pero como simplificación
        // añadimos atributos genéricos
        List<String> atributosClaseAsoc = new ArrayList<>();
        atributosClaseAsoc.add("extremo1:Object"); // Referencia genérica
        atributosClaseAsoc.add("extremo2:Object"); // Referencia genérica

        // Nota: En una implementación completa, se identificarían las clases extremas
        // reales
    }

    private static Asociacion parsearAsociacion(String asocStr) {
        // Formato esperado:
        // "nombreAtributo:tipoDestino:multiplicidad:unique_flag:sorted_flag"
        String[] partes = asocStr.split(":");
        if (partes.length >= 3) {
            String nombreAtributo = partes[0].trim();
            String tipoDestino = partes[1].trim();
            String multiplicidad = partes[2].trim();
            boolean unique = partes.length > 3 ? "unique".equals(partes[3].trim()) : false;
            boolean sorted = partes.length > 4 ? "sorted".equals(partes[4].trim()) : false;

            return new Asociacion(nombreAtributo, tipoDestino, multiplicidad, unique, sorted);
        }
        return null;
    }

    private static Operacion parsearOperacion(String opStr) {
        // Formato esperado:
        // "nombre:tipoRetorno[:parametros][:pre:expresion][:post:expresion]"
        String[] partes = opStr.split(":");
        if (partes.length >= 2) {
            String nombre = partes[0].trim();
            String tipoRetorno = partes[1].trim();
            if ("void".equals(tipoRetorno))
                tipoRetorno = null;

            List<String> parametros = new ArrayList<>();
            String precondicion = null;
            String postcondicion = null;

            // Parsear partes adicionales
            for (int i = 2; i < partes.length; i++) {
                String parte = partes[i].trim();
                if (parte.startsWith("pre:") && i + 1 < partes.length) {
                    precondicion = partes[i + 1].trim();
                    i++; // Saltar la siguiente parte
                } else if (parte.startsWith("post:") && i + 1 < partes.length) {
                    postcondicion = partes[i + 1].trim();
                    i++; // Saltar la siguiente parte
                } else if (!parte.startsWith("pre:") && !parte.startsWith("post:")) {
                    // Asumir que es un parámetro
                    parametros.add(parte);
                }
            }

            ContextoOCL contextOCL = (precondicion != null || postcondicion != null)
                    ? new ContextoOCL(precondicion, postcondicion)
                    : null;

            return new Operacion(nombre, tipoRetorno, parametros, contextOCL);
        }
        return null;
    }

    private static String obtenerTipoAtributoAsociacion(Asociacion asoc) {
        if (asoc.multiplicidad.equals("1") || asoc.multiplicidad.equals("0..1")) {
            return asoc.tipoDestino;
        } else {
            // Determinar tipo de colección según características
            if (asoc.unique && asoc.sorted) {
                return "SortedSet<" + asoc.tipoDestino + ">";
            } else if (asoc.unique && !asoc.sorted) {
                return "Set<" + asoc.tipoDestino + ">";
            } else if (!asoc.unique && !asoc.sorted) {
                return "AbstractCollection<" + asoc.tipoDestino + ">";
            } else { // !unique && sorted
                return "AbstractSequentialList<" + asoc.tipoDestino + ">";
            }
        }
    }

    private static boolean requiereInicializacion(Asociacion asoc) {
        return !"1".equals(asoc.multiplicidad) && !"0..1".equals(asoc.multiplicidad);
    }

    private static String obtenerImplementacionColeccion(Asociacion asoc) {
        if (asoc.unique && asoc.sorted) {
            return "TreeSet<" + asoc.tipoDestino + ">";
        } else if (asoc.unique && !asoc.sorted) {
            return "HashSet<" + asoc.tipoDestino + ">";
        } else if (!asoc.unique && !asoc.sorted) {
            return "ArrayList<" + asoc.tipoDestino + ">";
        } else { // !unique && sorted
            return "LinkedList<" + asoc.tipoDestino + ">";
        }
    }

    private static void generarMetodoOperacion(FileWriter writer, Operacion op) throws IOException {
        String tipoRetorno = op.tipoRetorno == null ? "void"
                : TYPE_MAPPING.getOrDefault(op.tipoRetorno, op.tipoRetorno);

        writer.write("    public " + tipoRetorno + " " + op.nombre + "(");

        // Parámetros
        List<String> parametrosFormateados = new ArrayList<>();
        for (String param : op.parametros) {
            String[] partes = param.split(" ");
            if (partes.length == 2) {
                String tipoParam = TYPE_MAPPING.getOrDefault(partes[0], partes[0]);
                parametrosFormateados.add(tipoParam + " " + partes[1]);
            } else {
                // Formato simple "tipo"
                String tipoParam = TYPE_MAPPING.getOrDefault(param, param);
                parametrosFormateados.add(tipoParam + " param" + (parametrosFormateados.size() + 1));
            }
        }
        writer.write(String.join(", ", parametrosFormateados));
        writer.write(") {\n");

        if (op.contextoOCL != null) {
            if (op.contextoOCL.precondicion != null) {
                writer.write("        // " + op.contextoOCL.precondicion + "\n");
            }
            writer.write("        // TODO\n");
            if (op.contextoOCL.postcondicion != null) {
                writer.write("        // " + op.contextoOCL.postcondicion + "\n");
            }
        } else {
            // Sin contexto OCL
            writer.write("        // TODO\n");
        }

        // Return por defecto si no es void
        if (!tipoRetorno.equals("void")) {
            String valorDefecto = switch (tipoRetorno) {
                case "boolean" -> "false";
                case "int", "float" -> "0";
                default -> "null";
            };
            writer.write("        return " + valorDefecto + ";\n");
        }

        writer.write("    }\n\n");
    }

    // Clases auxiliares para estructurar la información
    public static class Asociacion {
        public String nombreAtributo;
        public String tipoDestino;
        public String multiplicidad;
        public boolean unique;
        public boolean sorted;

        public Asociacion(String nombreAtributo, String tipoDestino, String multiplicidad, boolean unique,
                boolean sorted) {
            this.nombreAtributo = nombreAtributo;
            this.tipoDestino = tipoDestino;
            this.multiplicidad = multiplicidad;
            this.unique = unique;
            this.sorted = sorted;
        }
    }

    public static class Operacion {
        public String nombre;
        public String tipoRetorno;
        public List<String> parametros;
        public ContextoOCL contextoOCL;

        public Operacion(String nombre, String tipoRetorno, List<String> parametros, ContextoOCL contextoOCL) {
            this.nombre = nombre;
            this.tipoRetorno = tipoRetorno;
            this.parametros = parametros != null ? parametros : new ArrayList<>();
            this.contextoOCL = contextoOCL;
        }
    }

    public static class ContextoOCL {
        public String precondicion;
        public String postcondicion;

        public ContextoOCL(String precondicion, String postcondicion) {
            this.precondicion = precondicion;
            this.postcondicion = postcondicion;
        }
    }

}
