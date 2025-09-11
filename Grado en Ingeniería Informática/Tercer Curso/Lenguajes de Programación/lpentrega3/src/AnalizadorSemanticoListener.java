import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.antlr.v4.runtime.tree.TerminalNode;
import org.antlr.v4.runtime.Token;

public class AnalizadorSemanticoListener extends SimpleClassModelWithConstraintsBaseListener {

    private String currentEntity = null;
    private GeneradorCodigoJava.ContextoOCL currentContext = null;
    private Set<String> classNames;
    private Set<String> enumerationNames;
    private Map<String, List<String>> literalesPorEnumeracion;
    private Set<String> dataTypes;
    private List<String> semanticErrors;
    private Map<String, List<String>> invariantesPorClase;
    private Map<String, List<GeneradorCodigoJava.Asociacion>> asociacionesPorClase;
    public Map<String, List<GeneradorCodigoJava.Operacion>> operacionesPorEntidad;
    private List<DatatypeField> camposDatatype;
    private Map<String, List<Atributo>> atributosPorClase;
    private Map<String, Set<String>> targetsOpuestosPorClase = new HashMap<>();
    private Set<String> clasesAsociacion = null;

    // Tipos primitivos válidos
    private static final Set<String> PRIMITIVE_TYPES = Set.of("Boolean", "Integer", "Real", "String");

    private static class Atributo {
        String nombre;
        String tipo;
        int linea;
        String clase;

        Atributo(String nombre, String tipo, int linea, String clase) {
            this.nombre = nombre;
            this.tipo = tipo;
            this.linea = linea;
            this.clase = clase;
        }
    }

    private static class DatatypeField {
        String nombre;
        String tipo;
        String datatype;

        DatatypeField(String nombre, String tipo, String datatype) {
            this.nombre = nombre;
            this.tipo = tipo;
            this.datatype = datatype;
        }
    }

    public List<String> getSemanticErrors() {
        return semanticErrors;

    }

    public boolean hasErrors() {
        return !semanticErrors.isEmpty();
    }

    public Set<String> getClassNames() {
        return classNames;
    }

    public Set<String> getEnumerationNames() {
        return enumerationNames;
    }

    public Map<String, List<String>> getCamposPorDatatype() {
        Map<String, List<String>> mapa = new HashMap<>();
        for (DatatypeField campo : camposDatatype) {
            mapa.computeIfAbsent(campo.datatype, k -> new ArrayList<>())
                    .add(campo.nombre + ":" + campo.tipo);
        }
        return mapa;
    }

    public Map<String, List<String>> getAtributosPorClase() {
        Map<String, List<String>> mapa = new HashMap<>();

        for (Map.Entry<String, List<Atributo>> entry : atributosPorClase.entrySet()) {
            String clase = entry.getKey();
            List<Atributo> atributos = entry.getValue();

            for (Atributo a : atributos) {
                mapa.computeIfAbsent(clase, k -> new ArrayList<>())
                        .add(a.nombre + ":" + a.tipo);
            }
        }
        return mapa;
    }

    public Map<String, List<String>> getInvariantesPorClase() {
        return invariantesPorClase;
    }

    public Map<String, List<GeneradorCodigoJava.Asociacion>> getAsociacionesPorClase() {
        return asociacionesPorClase;
    }

    public Map<String, List<GeneradorCodigoJava.Operacion>> getOperacionesPorClase() {
        return operacionesPorEntidad;
    }

    @Override
    public void enterModel(SimpleClassModelWithConstraintsParser.ModelContext ctx) {
        System.out.println("\n Iniciando análisis del modelo... \n ");
        enumerationNames = new HashSet<>();
        dataTypes = new HashSet<>();
        semanticErrors = new ArrayList<>();
        atributosPorClase = new HashMap<>();
        camposDatatype = new ArrayList<>();
        asociacionesPorClase = new HashMap<>();
        operacionesPorEntidad = new HashMap<>();
        invariantesPorClase = new HashMap<>();
        targetsOpuestosPorClase = new HashMap<>();
        clasesAsociacion = new HashSet<>();
        literalesPorEnumeracion = new HashMap<>();

    }

    @Override
    public void enterClassModelSpecification(SimpleClassModelWithConstraintsParser.ClassModelSpecificationContext ctx) {
        classNames = new HashSet<>();
    }

    @Override
    public void enterClassDefinition(SimpleClassModelWithConstraintsParser.ClassDefinitionContext ctx) {
        String className = ctx.identifier(0).getText();
        currentEntity = ctx.identifier(0).getText();

        atributosPorClase.putIfAbsent(currentEntity, new ArrayList<>());
        targetsOpuestosPorClase.putIfAbsent(currentEntity, new HashSet<>());

        if (classNames.contains(className)) {
            System.err.println(
                    "Error semántico: Clase '" + className + "' ya está definida. Línea: " + ctx.start.getLine());
        } else {
            classNames.add(className);
            operacionesPorEntidad.put(className, new ArrayList<>());
            asociacionesPorClase.put(className, new ArrayList<>());
            System.out.println("Clase encontrada: " + className);
        }

        if (ctx.getChildCount() > 2 && ctx.getChild(2).getText().equals("extends")) {
            String parentClass = ctx.identifier(1).getText();
            System.out.println(" -> Hereda de: " + parentClass);

        }

        // Interfaces implementadas
        if (ctx.idList() != null) {
            List<SimpleClassModelWithConstraintsParser.IdentifierContext> interfaces = ctx.idList().identifier();
            System.out.print(" -> Implementa interfaces: ");
            for (SimpleClassModelWithConstraintsParser.IdentifierContext iface : interfaces) {
                System.out.print(iface.getText() + " ");
            }
            System.out.println();
        }
    }

    @Override
    public void enterOperationDefinition(SimpleClassModelWithConstraintsParser.OperationDefinitionContext ctx) {

        String nombreOperacion = ctx.identifier().getText();
        String tipoRetorno = ctx.type().getText();

        List<String> tiposParametros = new ArrayList<>();
        if (ctx.parameterDeclarations() != null) {
            for (SimpleClassModelWithConstraintsParser.ParameterDeclarationContext param : ctx.parameterDeclarations()
                    .parameterDeclaration()) {
                tiposParametros.add(param.type().getText());
            }
        }

        GeneradorCodigoJava.Operacion op = new GeneradorCodigoJava.Operacion(nombreOperacion, tipoRetorno,
                tiposParametros, currentContext);

        if (currentEntity != null && operacionesPorEntidad.containsKey(currentEntity)) {
            operacionesPorEntidad.get(currentEntity).add(op);
        }

    }

    @Override
    public void enterPrepostContext(SimpleClassModelWithConstraintsParser.PrepostContextContext ctx) {
        String nombreClase = ctx.ID().getText();
        var decl = ctx.operationDeclaration();
        String nombreOperacion = decl.identifier().getText();
        String tipoRetorno = decl.ocltype().getText();
        boolean encontrada = false;

        if (!classNames.contains(nombreClase)) {
            semanticErrors.add("Error: La clase '" + nombreClase + "' no está definida. Línea " + ctx.start.getLine());
            return;
        }

        List<String> tiposParametros = new ArrayList<>();
        if (decl.parameterDeclarations() != null) {
            for (SimpleClassModelWithConstraintsParser.ParameterDeclarationContext param : decl.parameterDeclarations()
                    .parameterDeclaration()) {
                tiposParametros.add(param.identifier().getText());
            }
        }

        for (GeneradorCodigoJava.Operacion op : operacionesPorEntidad.get(nombreClase)) {
            if (op.nombre.equals(nombreOperacion)
                    && op.tipoRetorno.equals(tipoRetorno)
                    && op.parametros.equals(tiposParametros)) {
                encontrada = true;

                String precondiciones = "";
                String postcondiciones = "";

                if (ctx.prepostSpecification() != null) {
                    // Unir todas las precondiciones en un solo string
                    precondiciones = ctx.prepostSpecification().precondition()
                            .stream()
                            .map(p -> p.expression().getText()) // extraer solo la expresión, sin 'pre:'
                            .collect(Collectors.joining("\n")); // o usa otro separador, como "; "

                    // Unir todas las postcondiciones en un solo string
                    postcondiciones = ctx.prepostSpecification().postcondition()
                            .stream()
                            .map(p -> p.expression().getText()) // extraer solo la expresión, sin 'post:'
                            .collect(Collectors.joining("\n"));
                }

                // Actualizar el contexto OCL de la operación
                op.contextoOCL = new GeneradorCodigoJava.ContextoOCL(precondiciones, postcondiciones);
                break;
            }
        }

        if (!encontrada) {
            semanticErrors.add("Error semantico (línea: " + +ctx.start.getLine() + "): La operación '" + nombreOperacion
                    + "' con firma (" + tiposParametros + ") : " + tipoRetorno +
                    " no está definida en la clase '" + nombreClase);
        }
    }

    @Override
    public void exitClassDefinition(SimpleClassModelWithConstraintsParser.ClassDefinitionContext ctx) {
        currentEntity = null;

    }

@Override
public void enterEnumeration(SimpleClassModelWithConstraintsParser.EnumerationContext ctx) {
    String enumName = ctx.identifier().getText();
    enumerationNames.add(enumName);

    List<String> literales = new ArrayList<>();
    for (SimpleClassModelWithConstraintsParser.EnumerationLiteralContext literalCtx : ctx.enumerationLiteral()) {
        literales.add(literalCtx.identifier().getText());
    }

    literalesPorEnumeracion.put(enumName, literales);
}


    @Override
    public void enterDatatypeDefinition(SimpleClassModelWithConstraintsParser.DatatypeDefinitionContext ctx) {
        String dtName = ctx.identifier().getText();
        dataTypes.add(dtName);
        operacionesPorEntidad.put(dtName, new ArrayList<>());
        currentEntity = dtName;
    }

    @Override
    public void exitDatatypeDefinition(SimpleClassModelWithConstraintsParser.DatatypeDefinitionContext ctx) {
        currentEntity = null;
    }

    @Override
    public void enterDatatypeBodyElement(SimpleClassModelWithConstraintsParser.DatatypeBodyElementContext ctx) {
        if (ctx.getChildCount() >= 4 && ctx.getChild(0).getText().equals("field")) {
            String nombreCampo = ctx.identifier().getText();
            String tipo = ctx.type().getText();
            camposDatatype.add(new DatatypeField(nombreCampo, tipo, currentEntity));
        }
    }

    @Override
    public void enterAttributeDefinition(SimpleClassModelWithConstraintsParser.AttributeDefinitionContext ctx) {
        String attrName = ctx.identifier().getText();
        String typeName = ctx.type().getText();
        int line = ctx.getStart().getLine();
        String className = currentEntity;

        Atributo nuevoAtributo = new Atributo(attrName, typeName, line, className);
        atributosPorClase.putIfAbsent(className, new ArrayList<>());
        atributosPorClase.get(className).add(nuevoAtributo);
    }

    @Override
    public void enterInvariantContext(SimpleClassModelWithConstraintsParser.InvariantContextContext ctx) {
        currentEntity = ctx.ID().getText();
    }

    @Override
    public void enterAssociation(SimpleClassModelWithConstraintsParser.AssociationContext ctx) {
        SimpleClassModelWithConstraintsParser.AssociationNameContext nameCtx = ctx.associationName();
        String associClassName = null; // Inicializar con valor por defecto

        if (nameCtx == null) {
            int line = ctx.getStart().getLine();
            semanticErrors.add("Error semántico (Línea " + line + "): La asociación entre '"
                    + ctx.associationEndA.identifier(0).getText() + "' y '"
                    + ctx.associationEndB.identifier(0).getText()
                    + "' no tiene nombre. Asigne un nombre con 'name = ID'.");
            return; // Detener procesamiento si no hay nombre
        }

        associClassName = nameCtx.identifier().getText();
        SimpleClassModelWithConstraintsParser.AssociationEndContext endA = ctx.associationEndA;

        SimpleClassModelWithConstraintsParser.AssociationEndContext endB = ctx.associationEndB;

        String targetA = endA.identifier(0).getText();
        String targetB = endB.identifier(0).getText();

        // Validar que las clases destino existan
        List<SimpleClassModelWithConstraintsParser.AssociationEndContext> targets = List.of(endA, endB);
        for (SimpleClassModelWithConstraintsParser.AssociationEndContext target : targets) {
            String className = target.identifier(0).getText();
            int line = target.getStart().getLine();

            if (!classNames.contains(className)) {
                semanticErrors.add("Error semántico (línea " + line + "): La clase destino '" +
                        className + "' de la asociación no está definida.");
            }
        }

        String roleB = endB.identifier().size() > 1 ? endB.identifier(1).getText() : "";

        String multiplicityB = endB.multiplicity() != null ? endB.multiplicity().getText() : "1";

        boolean uniqueB = false, sortedB = false;

        if (endB.constraints() != null) {
            String constraintsB = endB.constraints().getText();
            uniqueB = constraintsB.contains("unique");
            sortedB = constraintsB.contains("ordered");
        }

        String nombreRolB = !roleB.isEmpty() ? roleB : targetB.toLowerCase();
        GeneradorCodigoJava.Asociacion asociacionAB = new GeneradorCodigoJava.Asociacion(
                nombreRolB, targetB, multiplicityB, uniqueB, sortedB);

        asociacionesPorClase.computeIfAbsent(targetA, k -> new ArrayList<>()).add(asociacionAB);

        // Agregar la clase de asociación si es necesario (para asociaciones con
        // atributos)
        if (!classNames.contains(associClassName)) {
            classNames.add(associClassName);
            operacionesPorEntidad.put(associClassName, new ArrayList<>());
            asociacionesPorClase.put(associClassName, new ArrayList<>());
            System.out.println(" -> Clase de asociación reconocida: " + associClassName);
        }
        clasesAsociacion.add(associClassName);
        System.out.println(
                " -> Asociación creada: " + targetA + " <-> " + targetB + " (nombre: " + associClassName + ")");
    }

    @Override
    public void exitAssociation(SimpleClassModelWithConstraintsParser.AssociationContext ctx) {
        if (semanticErrors.isEmpty()) {
            SimpleClassModelWithConstraintsParser.AssociationEndContext endA = ctx.associationEndA;
            SimpleClassModelWithConstraintsParser.AssociationEndContext endB = ctx.associationEndB;

            String targetA = endA.identifier(0).getText();
            String targetB = endB.identifier(0).getText();

            if (!classNames.contains(targetA) || !classNames.contains(targetB)) {
                return; // Evitar procesamiento si hay clases inválidas
            }

            String roleA = endA.identifier().size() > 1 ? endA.identifier(1).getText() : "";
            String roleB = endB.identifier().size() > 1 ? endB.identifier(1).getText() : "";

            if (!roleA.isEmpty()) {
                targetsOpuestosPorClase.computeIfAbsent(targetB, k -> new HashSet<>()).add(roleA);
            }
            if (!roleB.isEmpty()) {
                targetsOpuestosPorClase.computeIfAbsent(targetA, k -> new HashSet<>()).add(roleB);
            }
        }
    }

    @Override
    public void exitAssociationEnd(SimpleClassModelWithConstraintsParser.AssociationEndContext ctx) {
        if (ctx.constraints() != null && ctx.multiplicity() != null) {
            String multiplicidad = ctx.multiplicity().getText();
            String restricciones = ctx.constraints().getText();

            if (multiplicidad.equals("1") || multiplicidad.equals("0..1")) {
                if (restricciones.contains("unique") || restricciones.contains("ordered")) {
                    String targetName = ctx.identifier(0).getText();
                    semanticErrors.add("Error: (línea: " + ctx.start.getLine()
                            + "):  No se pueden usar restricciones 'unique' u 'ordered' " +
                            "cuando la multiplicidad es " + multiplicidad +
                            " en la asociación hacia " + targetName);
                }
            }

            if (multiplicidad.contains("..")) {
                String[] partes = multiplicidad.split("\\.\\.");
                if (partes.length == 2 && !partes[1].equals("*")) {
                    int min = Integer.parseInt(partes[0]);
                    int max = Integer.parseInt(partes[1]);
                    if (min > max) {
                        semanticErrors.add("Error UML.7: Multiplicidad inválida '" + multiplicidad + "'");
                    }
                }
            }
        }
    }

    @Override
    public void enterInvariant(SimpleClassModelWithConstraintsParser.InvariantContext ctx) {
        if (currentEntity == null)
            return;
        String texto = ctx.getText();
        invariantesPorClase.computeIfAbsent(currentEntity, k -> new ArrayList<>()).add(texto);
    }

    /**
     * Método privado que valida un tipo de dato
     * 
     * @param typeName nombre del tipo de dato
     * @return true si está almacenado, false en caso contrario
     */
    private boolean isValidType(String typeName) {
        return PRIMITIVE_TYPES.contains(typeName)
                || enumerationNames.contains(typeName)
                || dataTypes.contains(typeName);
    }

    @Override
    public void exitInvariantContext(SimpleClassModelWithConstraintsParser.InvariantContextContext ctx) {

        String contextClass = ctx.ID().getText();
        if (!classNames.contains(contextClass)) {
            semanticErrors.add("Error: (línea: " + ctx.start.getLine() + "):  Clase '" + contextClass
                    + "' usada en un contexto OCL no ha sido definida. Línea: " + ctx.start.getLine());
        }
        currentEntity = null;
    }

    @Override
    public void exitBasicExpression(SimpleClassModelWithConstraintsParser.BasicExpressionContext ctx) {

        if (currentEntity != null && ctx.getText().startsWith("self.")) {
            String texto = ctx.getText();
            int posPunto = texto.indexOf(".");

            if (posPunto >= 0 && !texto.contains("(")) {
                String id = texto.substring(posPunto + 1);
                boolean esAtributo = atributosPorClase.containsKey(currentEntity) &&
                        atributosPorClase.get(currentEntity).contains(id);
                boolean esTargetOpuesto = targetsOpuestosPorClase.containsKey(currentEntity) &&
                        targetsOpuestosPorClase.get(currentEntity).contains(id);

                if (!esAtributo && !esTargetOpuesto) {
                    semanticErrors.add("Error (línea: " + ctx.start.getLine() + "):  En la clase '" + currentEntity +
                            "', el identificador '" + id +
                            "' en 'self." + id +
                            "' no es un atributo válido ni un target opuesto en una asociación.");
                }
            }
        }
    }

    /**
     * Override que sirve para depurar errores
     */
    @Override
    public void exitModel(SimpleClassModelWithConstraintsParser.ModelContext ctx) {

        System.out.print("\n ENUMS: ");
        for (String name : enumerationNames) {
            System.out.print(" " + name);
        }
        System.out.print("\n DATATYPES: ");
        for (String name : dataTypes) {
            System.out.print(" " + name);
        }
        System.out.print("\n CLASES: ");
        for (String name : classNames) {
            System.out.print(" " + name);
        }
        for (Map.Entry<String, List<Atributo>> entry : atributosPorClase.entrySet()) {
            List<Atributo> atributos = entry.getValue();

            for (Atributo a : atributos) {
                if (!isValidType(a.tipo)) {
                    semanticErrors.add("Error semántico (línea " + a.linea + "): El atributo '" +
                            a.nombre + "' tiene un tipo no permitido: '" + a.tipo + "'");
                }
            }
        }

    }

    public Map<String, List<String>> getLiteralesPorEnum() {
        return literalesPorEnumeracion;
    }

    public Set<String> getClasesAsociacion() {
        return clasesAsociacion;
    }

}
