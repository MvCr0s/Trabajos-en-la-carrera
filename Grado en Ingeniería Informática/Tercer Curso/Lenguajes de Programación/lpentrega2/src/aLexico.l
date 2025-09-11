%{
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "aSintactico.tab.h"
#define OPERATOR 1000

extern void yyerror(const char *s);
extern int yylineno;
extern YYSTYPE yylval;
int columna_actual = 1;

char* safe_strdup(const char* s) {
    char* new_s = strdup(s);
    if (!new_s) {
        perror("strdup failed");
        exit(EXIT_FAILURE);
    }
    return new_s;
}

typedef struct _simbolo {
    struct _simbolo *sig;
    char nombre[50];
    int valor;
    int tipo;
    int linea;
    int columna;
    char context[50]; // Nombre de la clase en la que se encuentra
} simbolo;

simbolo *tabla = NULL;
char contexto_actual[50] = "";
int contador_operaciones_coleccion = 0;

void insertar(simbolo **pT, simbolo *s) {
    s->sig = (*pT);
    (*pT) = s;
}

simbolo *buscar(simbolo *tabla, char nombre[20]) {
    while ((tabla != NULL) && (strcmp(nombre, tabla->nombre) != 0))
        tabla = tabla->sig;
    return tabla;
}
void imprimir_estadisticas(simbolo *tabla) {
    printf("\nEstadísticas por clase:\n");

    simbolo *temp = tabla;
    char clases[100][50];
    int conteo_clases = 0;

    while (temp) {
        int encontrado = 0;
        for (int i = 0; i < conteo_clases; i++) {
            if (strcmp(clases[i], temp->context) == 0) {
                encontrado = 1;
                break;
            }
        }
        if (!encontrado && strlen(temp->context) > 0) {
            strcpy(clases[conteo_clases++], temp->context);
        }
        temp = temp->sig;
    }

    printf("Clases con restricciones: %d\n\n", conteo_clases);

   for (int i = 0; i < conteo_clases; i++) {
        int invariantes = 0, operaciones = 0;
        temp = tabla;
        while (temp) {
            if (strcmp(temp->context, clases[i]) == 0) {
                if (temp->tipo == INV) invariantes++;
                if (temp->tipo == COLLECTION_OP) operaciones++;
            }
            temp = temp->sig;
        }
        printf("Clase: %s\n", clases[i]);
        printf("\tRestricciones: %d\n", invariantes);
        printf("\tOperaciones sobre colecciones: %d\n", operaciones);
    }
    printf("\nTotal de operaciones sobre colecciones: %d\n", contador_operaciones_coleccion);
}

void procesarToken(simbolo **tabla, char nombre[100], int tipo, int linea, int columna) {
    simbolo *nuevo = (simbolo *) malloc(sizeof(simbolo));
    strcpy(nuevo->nombre, nombre);
    nuevo->valor = (tipo == INT_LITERAL) ? atoi(nombre) : 0;
    nuevo->tipo = tipo;
    nuevo->linea = linea;
    nuevo->columna = columna;
    strcpy(nuevo->context, contexto_actual);
    insertar(tabla, nuevo);

    if (tipo == COLLECTION_OP) {
        contador_operaciones_coleccion++;
    }
    if (tipo == CONTEXT) {
        sscanf(yytext + 8, "%s", contexto_actual);
    }
}

%}

let [a-zA-Z]
dig [0-9]
digs {dig}+
id ({let}|{dig})({let}|{dig}|"_")*
frac [.,]{digs}
expo    [eE][+-]?{digs}
float {digs}{frac}?{expo}?
espacio [ \t\r\f]+
newline (\n|\r\n)

/* Expresiones Regulares OCL */
collection_op "->"(size|sum|max|min|isEmpty|notEmpty|includes|append|collect|select|reject|forAll|exists|iterate|asSequence|asBag|asSet|first|last|reverse|count|oclIsKindOf|oclIsTypeOf|oclAsType|flatten)
logic_op    and|or|xor|implies
rel_op      "<"|"<="|">="|">"|"="|"<>"
arith_op_multi abs|mod|div

%%
    /* Palabras Clave */
[Cc][Oo][Nn][Tt][Ee][Xx][Tt]":"? { columna_actual += yyleng; procesarToken(&tabla, yytext, CONTEXT, yylineno, columna_actual); return CONTEXT; }
[Ii][Nn][Vv]                  { columna_actual += yyleng; procesarToken(&tabla, yytext, INV, yylineno, columna_actual); return INV; }
[Pp][Rr][Ee]                  { columna_actual += yyleng; procesarToken(&tabla, yytext, PRE, yylineno, columna_actual); return PRE; }
[Pp][Oo][Ss][Tt]              { columna_actual += yyleng; procesarToken(&tabla, yytext, POST, yylineno, columna_actual); return POST; }
[Dd][Ee][Ff]                  { columna_actual += yyleng; procesarToken(&tabla, yytext, DEF, yylineno, columna_actual); return DEF; }
[Ll][Ee][Tt]                  { columna_actual += yyleng; procesarToken(&tabla, yytext, LET, yylineno, columna_actual); return LET; }
[Ii][Nn]                      { columna_actual += yyleng; procesarToken(&tabla, yytext, IN, yylineno, columna_actual); return IN; }
[Ii][Ff]                      { columna_actual += yyleng; procesarToken(&tabla, yytext, IF, yylineno, columna_actual); return IF; }
[Tt][Hh][Ee][Nn]              { columna_actual += yyleng; procesarToken(&tabla, yytext, THEN, yylineno, columna_actual); return THEN; }
[Ee][Ll][Ss][Ee]              { columna_actual += yyleng; procesarToken(&tabla, yytext, ELSE, yylineno, columna_actual); return ELSE; }
[Ee][Nn][Dd][Ii][Ff]          { columna_actual += yyleng; procesarToken(&tabla, yytext, ENDIF, yylineno, columna_actual); return ENDIF; }
[Ss][Ee][Ll][Ff]              { columna_actual += yyleng; procesarToken(&tabla, yytext, SELF, yylineno, columna_actual); return SELF; }
[Rr][Ee][Ss][Uu][Ll][Tt]      { columna_actual += yyleng; procesarToken(&tabla, yytext, RESULT, yylineno, columna_actual); return RESULT; }
[Tt][Rr][Uu][Ee]              { columna_actual += yyleng; yylval.bval = 1; procesarToken(&tabla, yytext, BOOLEAN_LITERAL, yylineno, columna_actual); return BOOLEAN_LITERAL; }
[Ff][Aa][Ll][Ss][Ee]          { columna_actual += yyleng; yylval.bval = 0; procesarToken(&tabla, yytext, BOOLEAN_LITERAL, yylineno, columna_actual); return BOOLEAN_LITERAL; }
[Nn][Uu][Ll][Ll]              { columna_actual += yyleng; procesarToken(&tabla, yytext, NULO, yylineno, columna_actual); return NULO; }
[Ss][Ee][Tt]                  { columna_actual += yyleng; procesarToken(&tabla, yytext, SET, yylineno, columna_actual); return SET; }
[Bb][Aa][Gg]                  { columna_actual += yyleng; procesarToken(&tabla, yytext, BAG, yylineno, columna_actual); return BAG; }
[Ss][Ee][Qq][Uu][Ee][Nn][Cc][Ee] { columna_actual += yyleng; procesarToken(&tabla, yytext, SEQUENCE, yylineno, columna_actual); return SEQUENCE; }
[Cc][Oo][Ll][Ll][Ee][Cc][Tt][Ii][Oo][Nn] { columna_actual += yyleng; procesarToken(&tabla, yytext, COLLECTION, yylineno, columna_actual); return COLLECTION; }
[Ii][Nn][Tt][Ee][Gg][Ee][Rr]  { columna_actual += yyleng; procesarToken(&tabla, yytext, INTEGER_TYPE, yylineno, columna_actual); return INTEGER_TYPE; }
[Rr][Ee][Aa][Ll]              { columna_actual += yyleng; procesarToken(&tabla, yytext, REAL_TYPE, yylineno, columna_actual); return REAL_TYPE; }
[Bb][Oo][Oo][Ll][Ee][Aa][Nn]  { columna_actual += yyleng; procesarToken(&tabla, yytext, BOOLEAN_TYPE, yylineno, columna_actual); return BOOLEAN_TYPE; }
[Ss][Tt][Rr][Ii][Nn][Gg]      { columna_actual += yyleng; procesarToken(&tabla, yytext, STRING_TYPE, yylineno, columna_actual); return STRING_TYPE; }
[Oo][Cc][Ll][Aa][Nn][Yy]      { columna_actual += yyleng; procesarToken(&tabla, yytext, OCLANY_TYPE, yylineno, columna_actual); return OCLANY_TYPE; }
[Oo][Cc][Ll][Tt][Yy][Pp][Ee]  { columna_actual += yyleng; procesarToken(&tabla, yytext, OCLTYPE_TYPE, yylineno, columna_actual); return OCLTYPE_TYPE; }

    /* Operadores y Colecciones */
{collection_op} { yylval.sval = safe_strdup(yytext); columna_actual += yyleng; procesarToken(&tabla, yytext, COLLECTION_OP, yylineno, columna_actual); return COLLECTION_OP; }
{logic_op}    { yylval.sval = safe_strdup(yytext); columna_actual += yyleng; procesarToken(&tabla, yytext, LOGIC_OP, yylineno, columna_actual); return LOGIC_OP; }
not    { columna_actual += yyleng; return NOT; }
{arith_op_multi} { yylval.sval = safe_strdup(yytext); columna_actual += yyleng; procesarToken(&tabla, yytext, ARITH_OP, yylineno, columna_actual); return ARITH_OP; }
{rel_op}      { yylval.sval = safe_strdup(yytext); columna_actual += yyleng; procesarToken(&tabla, yytext, REL_OP, yylineno, columna_actual); return REL_OP; }



"&&"    { yylval.sval = safe_strdup(yytext); columna_actual += yyleng; procesarToken(&tabla, yytext, LOGIC_OP, yylineno, columna_actual); return LOGIC_OP; }

    /* Operadores Aritméticos */
"+"           { yylval.sval = safe_strdup(yytext); columna_actual++; procesarToken(&tabla, yytext, OPERATOR, yylineno, columna_actual); return PLUS; }
"-"           { yylval.sval = safe_strdup(yytext); columna_actual++; procesarToken(&tabla, yytext, OPERATOR, yylineno, columna_actual); return MINUS; }
"*"           { yylval.sval = safe_strdup(yytext); columna_actual++; procesarToken(&tabla, yytext, OPERATOR, yylineno, columna_actual); return MULT; }
"/"           { yylval.sval = safe_strdup(yytext); columna_actual++; procesarToken(&tabla, yytext, OPERATOR, yylineno, columna_actual); return DIV; }



 /* Literales Numéricos */
{digs}      { yylval.ival = atoi(yytext); columna_actual += yyleng; procesarToken(&tabla, yytext, INT_LITERAL, yylineno, columna_actual); return INT_LITERAL; }

{float} {
    char *text = strdup(yytext);
    for (int i = 0; text[i]; i++) if (text[i] == ',') text[i] = '.';
    yylval.dval = atof(text);
    free(text);
    columna_actual += yyleng;
    procesarToken(&tabla, yytext, REAL_LITERAL, yylineno, columna_actual);
    return REAL_LITERAL;
}

 /* Identificadores */
{id}          { yylval.sval = safe_strdup(yytext); columna_actual += yyleng; procesarToken(&tabla, yytext, ID, yylineno, columna_actual); return ID; }


 /* Literal de String */
\'([^'\n]|\'\')*\' { yylval.sval = safe_strdup(yytext); columna_actual += strlen(yytext); procesarToken(&tabla, yytext, STRING_LITERAL, yylineno, columna_actual); return STRING_LITERAL; }
\"([^\"\n]|\"\")*\" { yylval.sval = safe_strdup(yytext); columna_actual += strlen(yytext); procesarToken(&tabla, yytext, STRING_LITERAL, yylineno, columna_actual); return STRING_LITERAL; }
“([^“”\n])*” { yylval.sval = safe_strdup(yytext); columna_actual += strlen(yytext); procesarToken(&tabla, yytext, STRING_LITERAL, yylineno, columna_actual); return STRING_LITERAL; }
‘([^‘’\n])*’ { yylval.sval = safe_strdup(yytext); columna_actual += strlen(yytext); procesarToken(&tabla, yytext, STRING_LITERAL, yylineno, columna_actual); return STRING_LITERAL; }
’([^’\n])*’ { yylval.sval = safe_strdup(yytext); columna_actual += strlen(yytext); procesarToken(&tabla, yytext, STRING_LITERAL, yylineno, columna_actual); return STRING_LITERAL; }

 /* Símbolos OCL */
"."           { columna_actual++; procesarToken(&tabla, yytext, DOT, yylineno, columna_actual); return DOT; }
":"           { columna_actual++; procesarToken(&tabla, yytext, COLON, yylineno, columna_actual); return COLON; }
"::"          { columna_actual += 2; procesarToken(&tabla, yytext, DCOLON, yylineno, columna_actual); return DCOLON; }
","           { columna_actual++; procesarToken(&tabla, yytext, COMA, yylineno, columna_actual); return COMA; }
"("           { columna_actual++; procesarToken(&tabla, yytext, LPAREN, yylineno, columna_actual); return LPAREN; }
")"           { columna_actual++; procesarToken(&tabla, yytext, RPAREN, yylineno, columna_actual); return RPAREN; }
"{"           { columna_actual++; procesarToken(&tabla, yytext, LBRACE, yylineno, columna_actual); return LBRACE; }
"}"           { columna_actual++; procesarToken(&tabla, yytext, RBRACE, yylineno, columna_actual); return RBRACE; }
"["           { columna_actual++; procesarToken(&tabla, yytext, LBRACKET, yylineno, columna_actual); return LBRACKET; }
"]"           { columna_actual++; procesarToken(&tabla, yytext, RBRACKET, yylineno, columna_actual); return RBRACKET; }
"|"           { columna_actual++; procesarToken(&tabla, yytext, BARRA, yylineno, columna_actual); return BARRA; }
"@"           { columna_actual++; procesarToken(&tabla, yytext, AT_SIGN, yylineno, columna_actual); return AT_SIGN; }

\n {
    columna_actual = 1;
    yylineno++;
}

[ \t\r\f]+ { columna_actual += yyleng; }

"--"[^\n]*    { /* Ignorar comentarios */ }

. {
    printf("Carácter no reconocido: '%s' en línea %d, columna %d\n", yytext, yylineno, columna_actual);
    columna_actual++;
}


%%

int yywrap() {
    return 1;
}

