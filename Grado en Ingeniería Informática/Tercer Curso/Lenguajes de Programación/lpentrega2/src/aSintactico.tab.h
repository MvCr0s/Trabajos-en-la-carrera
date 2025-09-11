/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_ASINTACTICO_TAB_H_INCLUDED
# define YY_YY_ASINTACTICO_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    ID = 258,                      /* ID  */
    STRING_LITERAL = 259,          /* STRING_LITERAL  */
    COLLECTION_OP = 260,           /* COLLECTION_OP  */
    LOGIC_OP = 261,                /* LOGIC_OP  */
    REL_OP = 262,                  /* REL_OP  */
    ARITH_OP = 263,                /* ARITH_OP  */
    IMPLIES = 264,                 /* IMPLIES  */
    PLUS = 265,                    /* PLUS  */
    MINUS = 266,                   /* MINUS  */
    MULT = 267,                    /* MULT  */
    DIV = 268,                     /* DIV  */
    NOT = 269,                     /* NOT  */
    INT_LITERAL = 270,             /* INT_LITERAL  */
    REAL_LITERAL = 271,            /* REAL_LITERAL  */
    BOOLEAN_LITERAL = 272,         /* BOOLEAN_LITERAL  */
    CONTEXT = 273,                 /* CONTEXT  */
    INV = 274,                     /* INV  */
    PRE = 275,                     /* PRE  */
    POST = 276,                    /* POST  */
    DEF = 277,                     /* DEF  */
    LET = 278,                     /* LET  */
    IN = 279,                      /* IN  */
    IF = 280,                      /* IF  */
    THEN = 281,                    /* THEN  */
    ELSE = 282,                    /* ELSE  */
    ENDIF = 283,                   /* ENDIF  */
    SELF = 284,                    /* SELF  */
    RESULT = 285,                  /* RESULT  */
    NULO = 286,                    /* NULO  */
    SET = 287,                     /* SET  */
    BAG = 288,                     /* BAG  */
    SEQUENCE = 289,                /* SEQUENCE  */
    COLLECTION = 290,              /* COLLECTION  */
    INTEGER_TYPE = 291,            /* INTEGER_TYPE  */
    REAL_TYPE = 292,               /* REAL_TYPE  */
    BOOLEAN_TYPE = 293,            /* BOOLEAN_TYPE  */
    STRING_TYPE = 294,             /* STRING_TYPE  */
    OCLANY_TYPE = 295,             /* OCLANY_TYPE  */
    OCLTYPE_TYPE = 296,            /* OCLTYPE_TYPE  */
    DOT = 297,                     /* DOT  */
    COLON = 298,                   /* COLON  */
    DCOLON = 299,                  /* DCOLON  */
    COMA = 300,                    /* COMA  */
    LPAREN = 301,                  /* LPAREN  */
    RPAREN = 302,                  /* RPAREN  */
    LBRACE = 303,                  /* LBRACE  */
    RBRACE = 304,                  /* RBRACE  */
    LBRACKET = 305,                /* LBRACKET  */
    RBRACKET = 306,                /* RBRACKET  */
    BARRA = 307,                   /* BARRA  */
    AT_SIGN = 308,                 /* AT_SIGN  */
    OR = 309,                      /* OR  */
    XOR = 310,                     /* XOR  */
    AND = 311,                     /* AND  */
    UMINUS = 312                   /* UMINUS  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 41 "aSintactico.y"

    int ival;
    double dval;
    int bval;
    char *sval;

#line 128 "aSintactico.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_ASINTACTICO_TAB_H_INCLUDED  */
