%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int yylex();
void yyerror(const char *s);
extern int yylineno;
extern char *yytext;
extern int columna_actual;
int syntax_errors_count = 0;
#define MAX_SYNTAX_ERRORS 10

typedef struct OperationStats {
    char *name;
    int collectionOpsCount;
    struct OperationStats *next;
} OperationStats;

typedef struct ClassStats {
    char *name;
    int invariantCount;
    int invCollectionOpsCount;
    OperationStats *operations;
    struct ClassStats *next;
} ClassStats;

ClassStats *statsList = NULL;
ClassStats *currentClass = NULL;
OperationStats *currentOperation = NULL;
int currentContextIsInvariant = 0;
int currentContextIsOperationContext = 0;

ClassStats* findOrCreateClass(const char* name);
OperationStats* findOrCreateOperation(ClassStats* cls, const char* opName);
void printStatistics();
void freeStatistics();
void safeFree(void *ptr) { if (ptr) free(ptr); }
%}

%union {
    int ival;
    double dval;
    int bval;
    char *sval;
}

%token <sval> ID STRING_LITERAL COLLECTION_OP LOGIC_OP REL_OP ARITH_OP IMPLIES PLUS MINUS MULT DIV NOT
%token <ival> INT_LITERAL
%token <dval> REAL_LITERAL
%token <bval> BOOLEAN_LITERAL
%token CONTEXT INV PRE POST DEF LET IN IF THEN ELSE ENDIF SELF RESULT NULO
%token SET BAG SEQUENCE COLLECTION INTEGER_TYPE REAL_TYPE BOOLEAN_TYPE STRING_TYPE OCLANY_TYPE OCLTYPE_TYPE
%token DOT COLON DCOLON COMA LPAREN RPAREN LBRACE RBRACE LBRACKET RBRACKET BARRA AT_SIGN 

%left IMPLIES
%left OR XOR
%left AND
%nonassoc NOT
%nonassoc REL_OP
%left PLUS MINUS
%left MULT DIV
%left ARITH_OP
%right UMINUS
%left DOT COLLECTION_OP

%start OclFile

%%

OclFile
    : /* vacío */
    | OclFile ContextDeclaration
    | OclFile error
      {
        syntax_errors_count++;
        currentClass = NULL;
        currentOperation = NULL;
        currentContextIsInvariant = 0;
        currentContextIsOperationContext = 0;

        // Sincronizar hasta el próximo CONTEXT o fin de archivo (0).
        while (1) {
            int token = yylex();
            if (token == 0) {
                // Llegamos al EOF: ya no hay nada más que parsear.
                break;
            }
            if (token == CONTEXT) {
                // "Reinyectamos" el token CONTEXT en yychar para que Bison lo reconozca.
                yychar = token;
                break;
            }
            // Si no es CONTEXT, seguimos leyendo hasta encontrarlo o EOF
        }

        // Indica a Bison que se limpió el error
        yyerrok;
        // Importante: NO usar yyclearin aquí, para que no se pierda el CONTEXT.
      }
    ;

ContextDeclaration
    : CONTEXT context_specifier
        {
            currentClass = NULL;
            currentOperation = NULL;
            currentContextIsInvariant = 0;
            currentContextIsOperationContext = 0;
        }
    ;

context_specifier
    : ID opt_context_type
        {
            currentClass = findOrCreateClass($1);
            currentOperation = NULL;
            printf("DEBUG: Contexto CLASE: %s\n", $1);
            safeFree($1);
        }
      classifier_context_body

    | ID DCOLON ID LPAREN opt_param_list RPAREN opt_return_type
        {
            currentClass = findOrCreateClass($1);
            currentOperation = findOrCreateOperation(currentClass, $3);
            printf("DEBUG: Contexto OPERACIÓN: %s::%s\n", $1, $3);
            safeFree($1); safeFree($3);
        }
      operation_context_body
    ;

opt_context_type : | COLON ID { safeFree($2); };

classifier_context_body
    : /* vacío */
    | classifier_context_body constraint_or_definition
    | classifier_context_body error
      {
        syntax_errors_count++;
        fprintf(stderr, "Error en el cuerpo del Context (línea %d). Saltando...\n", yylineno);

        // Consumir tokens hasta la siguiente palabra clave que inicie
        // un constraint (p. ej. INV o DEF) o el siguiente CONTEXT o EOF
        while (1) {
            int tk = yylex();
            if (tk == INV || tk == DEF || tk == CONTEXT || tk == 0) {

                if (tk == CONTEXT) {
                    yychar = tk; 
                    break;
                }

                break;
            }
        }
        yyerrok;      }
    ;

constraint_or_definition
    : INV optional_constraint_name COLON
        {
            if (currentClass) currentContextIsInvariant = 1;
            printf("DEBUG: Parseando INV\n");
        }
      expression
        {
            if (currentClass) {
                currentClass->invariantCount++;
                currentContextIsInvariant = 0;
                printf("DEBUG: INV terminado. Invs: %d, Ops colección en INV: %d\n",
                       currentClass->invariantCount, currentClass->invCollectionOpsCount);
            }
        }
    | DEF optional_constraint_name COLON expression { printf("DEBUG: DEF procesado\n"); }
    ;

optional_constraint_name : | ID { safeFree($1); };

operation_context_body
    : /* vacío */
    | operation_context_body condition
    | operation_context_body error
      {
        syntax_errors_count++;
        fprintf(stderr, "Error en cuerpo de la operación (línea %d). Saltando...\n", yylineno);

        while (1) {
            int tk = yylex();
            if (tk == PRE || tk == POST || tk == CONTEXT || tk == 0) {
                if (tk == CONTEXT) {
                    yychar = tk;
                    break;
                }

                break;
            }
        }
        yyerrok;
      }
    ;

condition
    : PRE optional_constraint_name COLON
        {
            if (currentOperation) currentContextIsOperationContext = 1;
            printf("DEBUG: Parseando PRE\n");
        }
      expression
        {
            if (currentOperation) {
                currentContextIsOperationContext = 0;
                printf("DEBUG: PRE terminado. Ops colección: %d\n", currentOperation->collectionOpsCount);
            }
        }
    | POST optional_constraint_name COLON
        {
            if (currentOperation) currentContextIsOperationContext = 1;
            printf("DEBUG: Parseando POST\n");
        }
      expression
        {
            if (currentOperation) {
                currentContextIsOperationContext = 0;
                printf("DEBUG: POST terminado. Ops colección: %d\n", currentOperation->collectionOpsCount);
            }
        }
    ;

expression
    : NOT expression
    | expression LOGIC_OP expression     { safeFree($2); }
    | expression REL_OP expression       { safeFree($2); }
    | expression IMPLIES expression      { safeFree($2); }
    | expression PLUS expression
    | expression MINUS expression
    | expression MULT expression
    | expression DIV expression
    | expression ARITH_OP expression     { safeFree($2); }
    | MINUS expression                     %prec UMINUS
    | expr_condicional
    | expr_primaria
    ;

expr_primaria
    : '-' expr_primaria %prec UMINUS
    | literal
    | SELF
    | RESULT
    | ID { safeFree($1); }
    | ID DCOLON ID { safeFree($1); safeFree($3); }
    | expr_primaria DCOLON ID { safeFree($3); }
    | collection_literal
    | AT_SIGN ID { safeFree($2); }
    | LPAREN expression RPAREN

    // Operaciones aritméticas internas
    | expr_primaria MULT expr_primaria
    | expr_primaria DIV expr_primaria

    // Acceso a atributos o métodos sin argumentos: obj.attr o obj.method()
    | expr_primaria DOT ID { safeFree($3); }
    | expr_primaria DOT ID LPAREN opt_argument_list RPAREN { safeFree($3); }

    | expr_primaria ID LPAREN opt_argument_list RPAREN expr_suffix { safeFree($2); }

    // Funciones globales tipo now()
    | ID LPAREN opt_argument_list RPAREN {
    printf("DEBUG: Función global: %s\n", $1);
    safeFree($1);
}





    // Collection operations con iterador
    | expr_primaria COLLECTION_OP ID LPAREN opt_param_list BARRA expression RPAREN
        {
            if (currentContextIsInvariant && currentClass)
                currentClass->invCollectionOpsCount++;
            else if (currentContextIsOperationContext && currentOperation)
                currentOperation->collectionOpsCount++;
            printf("DEBUG: CollectionOp (con iterador): %s\n", $2);
            safeFree($2); safeFree($3);
        }

    // Collection operations con argumentos
    | expr_primaria COLLECTION_OP LPAREN opt_argument_list RPAREN
        {
            if (currentContextIsInvariant && currentClass)
                currentClass->invCollectionOpsCount++;
            else if (currentContextIsOperationContext && currentOperation)
                currentOperation->collectionOpsCount++;
            printf("DEBUG: CollectionOp (con argumentos): %s\n", $2);
            safeFree($2);
        }

    // Collection operations sin argumentos
    | expr_primaria COLLECTION_OP LPAREN RPAREN
        {
            if (currentContextIsInvariant && currentClass)
                currentClass->invCollectionOpsCount++;
            else if (currentContextIsOperationContext && currentOperation)
                currentOperation->collectionOpsCount++;
            printf("DEBUG: CollectionOp (sin args): %s\n", $2);
            safeFree($2);
        }

    | expr_primaria COLLECTION_OP LPAREN RPAREN REL_OP expression
{
    if (currentContextIsInvariant && currentClass)
        currentClass->invCollectionOpsCount++;
    else if (currentContextIsOperationContext && currentOperation)
        currentOperation->collectionOpsCount++;
    printf("DEBUG: CollectionOp + REL_OP: %s\n", $2);
    safeFree($2);
}

    ;


expr_suffix
    : /* vacío */
    | DOT ID expr_suffix { safeFree($2); }
    | DOT ID LPAREN opt_argument_list RPAREN expr_suffix { safeFree($2); }
    | LPAREN opt_argument_list RPAREN expr_suffix
;


expr_condicional
    : IF expression THEN expression ELSE expression ENDIF
    ;

literal
    : INT_LITERAL
    | REAL_LITERAL
    | STRING_LITERAL { safeFree($1); }
    | BOOLEAN_LITERAL
    | NULO
    ;

collection_literal
    : collection_type LBRACE RBRACE
    | collection_type LBRACE expression_list RBRACE
    ;

collection_type : SET | BAG | SEQUENCE | COLLECTION;
expression_list : expression | expression_list COMA expression;
opt_argument_list : | expression_list;
opt_param_list : | param_declaration_list;

param_declaration_list
    : param_declaration
    | param_declaration_list COMA param_declaration
    ;

param_declaration : ID opt_type_specifier { safeFree($1); };
opt_return_type : | COLON type;
opt_type_specifier : | COLON type;
type : primitive_type | ID { safeFree($1); };
primitive_type : INTEGER_TYPE | REAL_TYPE | BOOLEAN_TYPE | STRING_TYPE | OCLANY_TYPE | OCLTYPE_TYPE;

%%

ClassStats* findOrCreateClass(const char* name) {
    ClassStats *current = statsList;
    while (current) {
        if (strcmp(current->name, name) == 0) return current;
        current = current->next;
    }
    ClassStats *newClass = malloc(sizeof(ClassStats));
    newClass->name = strdup(name);
    newClass->invariantCount = 0;
    newClass->invCollectionOpsCount = 0;
    newClass->operations = NULL;
    newClass->next = statsList;
    statsList = newClass;
    return newClass;
}

OperationStats* findOrCreateOperation(ClassStats* cls, const char* opName) {
    if (!cls) return NULL;
    OperationStats *current = cls->operations;
    while (current) {
        if (strcmp(current->name, opName) == 0) return current;
        current = current->next;
    }
    OperationStats *newOp = malloc(sizeof(OperationStats));
    newOp->name = strdup(opName);
    newOp->collectionOpsCount = 0;
    newOp->next = cls->operations;
    cls->operations = newOp;
    return newOp;
}

void printStatistics() {
    ClassStats *cls = statsList;
    while (cls) {
        printf("nombre de la clase: %s\n", cls->name);
        printf("cantidad de invariantes de la clase: %d\n", cls->invariantCount);
        printf("cantidad de operaciones sobre colecciones utilizadas en los invariantes: %d\n", cls->invCollectionOpsCount);
        printf("Operaciones de la clase especificadas:\n");
        if (cls->operations) {
            OperationStats *op = cls->operations;
            while (op) {
                printf("    nombre de la operación: %s\n", op->name);
                printf("    cantidad de operaciones sobre colecciones utilizadas en la especificación: %d\n", op->collectionOpsCount);
                op = op->next;
            }
        } else {
            printf("    No se especificaron operaciones.\n");
        }
        cls = cls->next;
    }
}


void freeStatistics() {
    ClassStats *cls = statsList;
    while (cls) {
        OperationStats *op = cls->operations;
        while (op) {
            OperationStats *nextOp = op->next;
            safeFree(op->name);
            free(op);
            op = nextOp;
        }
        ClassStats *nextCls = cls->next;
        safeFree(cls->name);
        free(cls);
        cls = nextCls;
    }
    statsList = NULL;
}

int main(int argc, char **argv) {
    extern FILE *yyin;
    if (argc > 1) {
        yyin = fopen(argv[1], "r");
        if (!yyin) { perror(argv[1]); return 1; }
    } else {
        printf("Uso: %s <archivo_ocl>\nLeyendo desde stdin...\n", argv[0]);
        yyin = stdin;
    }

    printf("Iniciando análisis sintáctico...\n");
    int parse_result = yyparse();

    if (parse_result == 0 && syntax_errors_count == 0) {
        printf("Análisis sintáctico completado con éxito.\n");
        printStatistics();
    } else {
        printf("Análisis sintáctico completado con errores (%d).\n", syntax_errors_count);
    }

    if (yyin != stdin) fclose(yyin);
    freeStatistics();
    return (syntax_errors_count == 0 ? 0 : 1);
}

void yyerror(const char *s) {
    fprintf(stderr, "Error en la línea %d: %s cerca de '%s'\n", yylineno, s, yytext);
    syntax_errors_count++;
    if (syntax_errors_count > MAX_SYNTAX_ERRORS) {
        fprintf(stderr, "Se alcanzó el máximo de errores permitidos. Abortando.\n");
        exit(1);
    }
}



