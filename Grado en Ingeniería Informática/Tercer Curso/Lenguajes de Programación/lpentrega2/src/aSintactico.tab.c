/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 1 "aSintactico.y"

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

#line 111 "aSintactico.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "aSintactico.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_ID = 3,                         /* ID  */
  YYSYMBOL_STRING_LITERAL = 4,             /* STRING_LITERAL  */
  YYSYMBOL_COLLECTION_OP = 5,              /* COLLECTION_OP  */
  YYSYMBOL_LOGIC_OP = 6,                   /* LOGIC_OP  */
  YYSYMBOL_REL_OP = 7,                     /* REL_OP  */
  YYSYMBOL_ARITH_OP = 8,                   /* ARITH_OP  */
  YYSYMBOL_IMPLIES = 9,                    /* IMPLIES  */
  YYSYMBOL_PLUS = 10,                      /* PLUS  */
  YYSYMBOL_MINUS = 11,                     /* MINUS  */
  YYSYMBOL_MULT = 12,                      /* MULT  */
  YYSYMBOL_DIV = 13,                       /* DIV  */
  YYSYMBOL_NOT = 14,                       /* NOT  */
  YYSYMBOL_INT_LITERAL = 15,               /* INT_LITERAL  */
  YYSYMBOL_REAL_LITERAL = 16,              /* REAL_LITERAL  */
  YYSYMBOL_BOOLEAN_LITERAL = 17,           /* BOOLEAN_LITERAL  */
  YYSYMBOL_CONTEXT = 18,                   /* CONTEXT  */
  YYSYMBOL_INV = 19,                       /* INV  */
  YYSYMBOL_PRE = 20,                       /* PRE  */
  YYSYMBOL_POST = 21,                      /* POST  */
  YYSYMBOL_DEF = 22,                       /* DEF  */
  YYSYMBOL_LET = 23,                       /* LET  */
  YYSYMBOL_IN = 24,                        /* IN  */
  YYSYMBOL_IF = 25,                        /* IF  */
  YYSYMBOL_THEN = 26,                      /* THEN  */
  YYSYMBOL_ELSE = 27,                      /* ELSE  */
  YYSYMBOL_ENDIF = 28,                     /* ENDIF  */
  YYSYMBOL_SELF = 29,                      /* SELF  */
  YYSYMBOL_RESULT = 30,                    /* RESULT  */
  YYSYMBOL_NULO = 31,                      /* NULO  */
  YYSYMBOL_SET = 32,                       /* SET  */
  YYSYMBOL_BAG = 33,                       /* BAG  */
  YYSYMBOL_SEQUENCE = 34,                  /* SEQUENCE  */
  YYSYMBOL_COLLECTION = 35,                /* COLLECTION  */
  YYSYMBOL_INTEGER_TYPE = 36,              /* INTEGER_TYPE  */
  YYSYMBOL_REAL_TYPE = 37,                 /* REAL_TYPE  */
  YYSYMBOL_BOOLEAN_TYPE = 38,              /* BOOLEAN_TYPE  */
  YYSYMBOL_STRING_TYPE = 39,               /* STRING_TYPE  */
  YYSYMBOL_OCLANY_TYPE = 40,               /* OCLANY_TYPE  */
  YYSYMBOL_OCLTYPE_TYPE = 41,              /* OCLTYPE_TYPE  */
  YYSYMBOL_DOT = 42,                       /* DOT  */
  YYSYMBOL_COLON = 43,                     /* COLON  */
  YYSYMBOL_DCOLON = 44,                    /* DCOLON  */
  YYSYMBOL_COMA = 45,                      /* COMA  */
  YYSYMBOL_LPAREN = 46,                    /* LPAREN  */
  YYSYMBOL_RPAREN = 47,                    /* RPAREN  */
  YYSYMBOL_LBRACE = 48,                    /* LBRACE  */
  YYSYMBOL_RBRACE = 49,                    /* RBRACE  */
  YYSYMBOL_LBRACKET = 50,                  /* LBRACKET  */
  YYSYMBOL_RBRACKET = 51,                  /* RBRACKET  */
  YYSYMBOL_BARRA = 52,                     /* BARRA  */
  YYSYMBOL_AT_SIGN = 53,                   /* AT_SIGN  */
  YYSYMBOL_OR = 54,                        /* OR  */
  YYSYMBOL_XOR = 55,                       /* XOR  */
  YYSYMBOL_AND = 56,                       /* AND  */
  YYSYMBOL_UMINUS = 57,                    /* UMINUS  */
  YYSYMBOL_58_ = 58,                       /* '-'  */
  YYSYMBOL_YYACCEPT = 59,                  /* $accept  */
  YYSYMBOL_OclFile = 60,                   /* OclFile  */
  YYSYMBOL_ContextDeclaration = 61,        /* ContextDeclaration  */
  YYSYMBOL_context_specifier = 62,         /* context_specifier  */
  YYSYMBOL_63_1 = 63,                      /* $@1  */
  YYSYMBOL_64_2 = 64,                      /* $@2  */
  YYSYMBOL_opt_context_type = 65,          /* opt_context_type  */
  YYSYMBOL_classifier_context_body = 66,   /* classifier_context_body  */
  YYSYMBOL_constraint_or_definition = 67,  /* constraint_or_definition  */
  YYSYMBOL_68_3 = 68,                      /* $@3  */
  YYSYMBOL_optional_constraint_name = 69,  /* optional_constraint_name  */
  YYSYMBOL_operation_context_body = 70,    /* operation_context_body  */
  YYSYMBOL_condition = 71,                 /* condition  */
  YYSYMBOL_72_4 = 72,                      /* $@4  */
  YYSYMBOL_73_5 = 73,                      /* $@5  */
  YYSYMBOL_expression = 74,                /* expression  */
  YYSYMBOL_expr_primaria = 75,             /* expr_primaria  */
  YYSYMBOL_expr_suffix = 76,               /* expr_suffix  */
  YYSYMBOL_expr_condicional = 77,          /* expr_condicional  */
  YYSYMBOL_literal = 78,                   /* literal  */
  YYSYMBOL_collection_literal = 79,        /* collection_literal  */
  YYSYMBOL_collection_type = 80,           /* collection_type  */
  YYSYMBOL_expression_list = 81,           /* expression_list  */
  YYSYMBOL_opt_argument_list = 82,         /* opt_argument_list  */
  YYSYMBOL_opt_param_list = 83,            /* opt_param_list  */
  YYSYMBOL_param_declaration_list = 84,    /* param_declaration_list  */
  YYSYMBOL_param_declaration = 85,         /* param_declaration  */
  YYSYMBOL_opt_return_type = 86,           /* opt_return_type  */
  YYSYMBOL_opt_type_specifier = 87,        /* opt_type_specifier  */
  YYSYMBOL_type = 88,                      /* type  */
  YYSYMBOL_primitive_type = 89             /* primitive_type  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   264

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  59
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  31
/* YYNRULES -- Number of rules.  */
#define YYNRULES  95
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  167

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   312


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,    58,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    72,    72,    73,    74,   104,   115,   114,   124,   123,
     133,   133,   136,   137,   138,   162,   161,   175,   178,   178,
     181,   182,   183,   205,   204,   217,   216,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   258,
     259,   262,   263,   265,   268,   278,   289,   300,   310,   324,
     325,   326,   327,   332,   336,   337,   338,   339,   340,   344,
     345,   348,   348,   348,   348,   349,   349,   350,   350,   351,
     351,   354,   355,   358,   359,   359,   360,   360,   361,   361,
     362,   362,   362,   362,   362,   362
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "ID", "STRING_LITERAL",
  "COLLECTION_OP", "LOGIC_OP", "REL_OP", "ARITH_OP", "IMPLIES", "PLUS",
  "MINUS", "MULT", "DIV", "NOT", "INT_LITERAL", "REAL_LITERAL",
  "BOOLEAN_LITERAL", "CONTEXT", "INV", "PRE", "POST", "DEF", "LET", "IN",
  "IF", "THEN", "ELSE", "ENDIF", "SELF", "RESULT", "NULO", "SET", "BAG",
  "SEQUENCE", "COLLECTION", "INTEGER_TYPE", "REAL_TYPE", "BOOLEAN_TYPE",
  "STRING_TYPE", "OCLANY_TYPE", "OCLTYPE_TYPE", "DOT", "COLON", "DCOLON",
  "COMA", "LPAREN", "RPAREN", "LBRACE", "RBRACE", "LBRACKET", "RBRACKET",
  "BARRA", "AT_SIGN", "OR", "XOR", "AND", "UMINUS", "'-'", "$accept",
  "OclFile", "ContextDeclaration", "context_specifier", "$@1", "$@2",
  "opt_context_type", "classifier_context_body",
  "constraint_or_definition", "$@3", "optional_constraint_name",
  "operation_context_body", "condition", "$@4", "$@5", "expression",
  "expr_primaria", "expr_suffix", "expr_condicional", "literal",
  "collection_literal", "collection_type", "expression_list",
  "opt_argument_list", "opt_param_list", "param_declaration_list",
  "param_declaration", "opt_return_type", "opt_type_specifier", "type",
  "primitive_type", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-77)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-10)

#define yytable_value_is_error(Yyn) \
  ((Yyn) == YYTABLE_NINF)

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     -77,     9,   -77,   -77,    -1,   -77,   -30,   -77,    17,    34,
     -77,   -77,     3,   -77,    62,   110,    20,    27,    36,   -77,
     -77,    83,    83,   -77,   124,   -77,    55,    62,   -77,    71,
      72,   -77,   -77,   -77,   -77,   -77,   -77,   -77,   -77,   -77,
     124,   -77,   -77,   -77,   122,   -77,   -77,   122,    11,   -77,
     122,   122,   -77,   -77,   -77,   122,   -77,   -77,   -77,   -77,
     -77,   -77,   -77,   122,   106,   155,   246,   137,   -77,   -77,
     -77,    68,    75,   246,   121,   122,   125,   199,   238,    60,
     -77,    18,   122,   122,   122,   122,   122,   122,   122,   122,
      84,     5,   155,   155,   140,   141,     1,   -77,    83,    83,
     -77,   -77,   246,   100,   101,   122,   -77,   246,   227,   125,
     199,    16,    16,    50,    50,   122,   120,    88,    18,    18,
     123,   -77,   -77,    38,   130,   131,   122,   -77,   216,   129,
      62,   160,   135,   122,   -77,   -77,   -77,   246,   122,    42,
     126,   122,   -77,   136,   122,   122,   208,   174,   122,   -77,
     122,   227,   -77,   246,   246,   -77,    66,   152,   185,   122,
     -77,    42,   -77,   153,   -77,    42,   -77
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       2,     0,     1,     4,     0,     3,    10,     5,     0,     0,
       6,    11,     0,    12,    79,     0,    86,     0,    80,    81,
      14,    18,    18,    13,     0,    83,    84,     0,    19,     0,
       0,    89,    90,    91,    92,    93,    94,    95,    87,    88,
       0,     8,    82,    15,     0,    85,    20,     0,    43,    66,
       0,     0,    64,    65,    67,     0,    41,    42,    68,    71,
      72,    73,    74,     0,     0,     0,    17,    38,    37,    40,
      46,     0,     0,    16,     0,    77,    36,    27,     0,     0,
      47,    39,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    22,    18,    18,
      21,    44,    75,    78,     0,     0,    48,    28,    29,    35,
      30,    31,    32,    33,    34,    77,     0,     0,    49,    50,
      51,    45,    69,     0,     0,     0,     0,    54,     0,     0,
      79,    57,     0,    77,    70,    23,    25,    76,     0,    59,
       0,     0,    56,     0,     0,     0,     0,     0,    77,    53,
       0,    58,    52,    24,    26,    63,    59,     0,     0,    77,
      60,    59,    55,     0,    62,    59,    61
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -77,   -77,   -77,   -77,   -77,   -77,   -77,   -77,   -77,   -77,
     -21,   -77,   -77,   -77,   -77,   -44,   -40,   -76,   -77,   -77,
     -77,   -77,   107,   -69,    74,   -77,   175,   -77,   -77,   190,
     -77
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_uint8 yydefgoto[] =
{
       0,     1,     5,     7,    13,    46,    10,    15,    23,    47,
      29,    72,   100,   144,   145,   102,    67,   149,    68,    69,
      70,    71,   103,   104,    17,    18,    19,    41,    25,    38,
      39
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      66,    30,     6,    73,    48,    49,    76,    77,   116,     2,
       3,    78,    50,     8,     9,    51,    52,    53,    54,    79,
      11,    90,    82,    91,    84,    81,    55,     4,    88,    89,
      56,    57,    58,    59,    60,    61,    62,    12,   107,   108,
     109,   110,   111,   112,   113,   114,   129,    63,   132,    14,
     122,   117,   118,   119,    64,    74,    82,    75,    84,    65,
      94,   128,    95,    24,   143,    16,    82,    83,    84,    85,
      86,    87,    88,    89,    26,    -9,    97,   124,   125,   157,
     160,    27,   137,   126,   147,   164,    28,   134,   148,   166,
     163,    48,    49,    -9,   146,    98,    99,   151,    40,    50,
     153,   154,    51,    52,    53,    54,   158,   106,   147,    80,
      -7,    20,   159,    55,    43,    44,    96,    56,    57,    58,
      59,    60,    61,    62,   101,    48,    49,    31,    -7,    21,
     115,    82,    22,    50,    63,   131,    51,    52,    53,    54,
      90,    64,    91,   120,   121,   126,    65,    55,   127,    92,
      93,    56,    57,    58,    59,    60,    61,    62,    48,    49,
      32,    33,    34,    35,    36,    37,   130,   141,    63,   133,
      52,    53,    54,   135,   136,    64,   139,   156,   150,    94,
      65,    95,   142,   152,    56,    57,    58,    59,    60,    61,
      62,    82,    83,    84,    85,    86,    87,    88,    89,   161,
     165,    63,    42,   123,   140,    82,    83,    84,    64,    86,
      87,    88,    89,    65,    82,    83,    84,    85,    86,    87,
      88,    89,    82,    83,    84,    85,    86,    87,    88,    89,
      45,     0,   162,    82,   -10,    84,   155,    86,    87,    88,
      89,     0,     0,   138,    82,    83,    84,    85,    86,    87,
      88,    89,    82,    83,    84,    85,    86,    87,    88,    89,
       0,     0,     0,     0,   105
};

static const yytype_int16 yycheck[] =
{
      44,    22,     3,    47,     3,     4,    50,    51,     3,     0,
       1,    55,    11,    43,    44,    14,    15,    16,    17,    63,
       3,     3,     6,     5,     8,    65,    25,    18,    12,    13,
      29,    30,    31,    32,    33,    34,    35,     3,    82,    83,
      84,    85,    86,    87,    88,    89,   115,    46,   117,    46,
      49,    46,    92,    93,    53,    44,     6,    46,     8,    58,
      42,   105,    44,    43,   133,     3,     6,     7,     8,     9,
      10,    11,    12,    13,    47,     0,     1,    98,    99,   148,
     156,    45,   126,    45,    42,   161,     3,    49,    46,   165,
     159,     3,     4,    18,   138,    20,    21,   141,    43,    11,
     144,   145,    14,    15,    16,    17,   150,    47,    42,     3,
       0,     1,    46,    25,    43,    43,    48,    29,    30,    31,
      32,    33,    34,    35,     3,     3,     4,     3,    18,    19,
      46,     6,    22,    11,    46,    47,    14,    15,    16,    17,
       3,    53,     5,     3,     3,    45,    58,    25,    47,    12,
      13,    29,    30,    31,    32,    33,    34,    35,     3,     4,
      36,    37,    38,    39,    40,    41,    46,     7,    46,    46,
      15,    16,    17,    43,    43,    53,    47,     3,    52,    42,
      58,    44,    47,    47,    29,    30,    31,    32,    33,    34,
      35,     6,     7,     8,     9,    10,    11,    12,    13,    47,
      47,    46,    27,    96,   130,     6,     7,     8,    53,    10,
      11,    12,    13,    58,     6,     7,     8,     9,    10,    11,
      12,    13,     6,     7,     8,     9,    10,    11,    12,    13,
      40,    -1,    47,     6,     7,     8,    28,    10,    11,    12,
      13,    -1,    -1,    27,     6,     7,     8,     9,    10,    11,
      12,    13,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    -1,    -1,    -1,    26
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    60,     0,     1,    18,    61,     3,    62,    43,    44,
      65,     3,     3,    63,    46,    66,     3,    83,    84,    85,
       1,    19,    22,    67,    43,    87,    47,    45,     3,    69,
      69,     3,    36,    37,    38,    39,    40,    41,    88,    89,
      43,    86,    85,    43,    43,    88,    64,    68,     3,     4,
      11,    14,    15,    16,    17,    25,    29,    30,    31,    32,
      33,    34,    35,    46,    53,    58,    74,    75,    77,    78,
      79,    80,    70,    74,    44,    46,    74,    74,    74,    74,
       3,    75,     6,     7,     8,     9,    10,    11,    12,    13,
       3,     5,    12,    13,    42,    44,    48,     1,    20,    21,
      71,     3,    74,    81,    82,    26,    47,    74,    74,    74,
      74,    74,    74,    74,    74,    46,     3,    46,    75,    75,
       3,     3,    49,    81,    69,    69,    45,    47,    74,    82,
      46,    47,    82,    46,    49,    43,    43,    74,    27,    47,
      83,     7,    47,    82,    72,    73,    74,    42,    46,    76,
      52,    74,    47,    74,    74,    28,     3,    82,    74,    46,
      76,    47,    47,    82,    76,    47,    76
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    59,    60,    60,    60,    61,    63,    62,    64,    62,
      65,    65,    66,    66,    66,    68,    67,    67,    69,    69,
      70,    70,    70,    72,    71,    73,    71,    74,    74,    74,
      74,    74,    74,    74,    74,    74,    74,    74,    74,    75,
      75,    75,    75,    75,    75,    75,    75,    75,    75,    75,
      75,    75,    75,    75,    75,    75,    75,    75,    75,    76,
      76,    76,    76,    77,    78,    78,    78,    78,    78,    79,
      79,    80,    80,    80,    80,    81,    81,    82,    82,    83,
      83,    84,    84,    85,    86,    86,    87,    87,    88,    88,
      89,    89,    89,    89,    89,    89
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     0,     2,     2,     2,     0,     4,     0,     9,
       0,     2,     0,     2,     2,     0,     5,     4,     0,     1,
       0,     2,     2,     0,     5,     0,     5,     2,     3,     3,
       3,     3,     3,     3,     3,     3,     2,     1,     1,     2,
       1,     1,     1,     1,     3,     3,     1,     2,     3,     3,
       3,     3,     6,     6,     4,     8,     5,     4,     6,     0,
       3,     6,     4,     7,     1,     1,     1,     1,     1,     3,
       4,     1,     1,     1,     1,     1,     3,     0,     1,     0,
       1,     1,     3,     2,     0,     2,     0,     2,     1,     1,
       1,     1,     1,     1,     1,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 4: /* OclFile: OclFile error  */
#line 75 "aSintactico.y"
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
#line 1353 "aSintactico.tab.c"
    break;

  case 5: /* ContextDeclaration: CONTEXT context_specifier  */
#line 105 "aSintactico.y"
        {
            currentClass = NULL;
            currentOperation = NULL;
            currentContextIsInvariant = 0;
            currentContextIsOperationContext = 0;
        }
#line 1364 "aSintactico.tab.c"
    break;

  case 6: /* $@1: %empty  */
#line 115 "aSintactico.y"
        {
            currentClass = findOrCreateClass((yyvsp[-1].sval));
            currentOperation = NULL;
            printf("DEBUG: Contexto CLASE: %s\n", (yyvsp[-1].sval));
            safeFree((yyvsp[-1].sval));
        }
#line 1375 "aSintactico.tab.c"
    break;

  case 8: /* $@2: %empty  */
#line 124 "aSintactico.y"
        {
            currentClass = findOrCreateClass((yyvsp[-6].sval));
            currentOperation = findOrCreateOperation(currentClass, (yyvsp[-4].sval));
            printf("DEBUG: Contexto OPERACIÓN: %s::%s\n", (yyvsp[-6].sval), (yyvsp[-4].sval));
            safeFree((yyvsp[-6].sval)); safeFree((yyvsp[-4].sval));
        }
#line 1386 "aSintactico.tab.c"
    break;

  case 11: /* opt_context_type: COLON ID  */
#line 133 "aSintactico.y"
                              { safeFree((yyvsp[0].sval)); }
#line 1392 "aSintactico.tab.c"
    break;

  case 14: /* classifier_context_body: classifier_context_body error  */
#line 139 "aSintactico.y"
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
#line 1416 "aSintactico.tab.c"
    break;

  case 15: /* $@3: %empty  */
#line 162 "aSintactico.y"
        {
            if (currentClass) currentContextIsInvariant = 1;
            printf("DEBUG: Parseando INV\n");
        }
#line 1425 "aSintactico.tab.c"
    break;

  case 16: /* constraint_or_definition: INV optional_constraint_name COLON $@3 expression  */
#line 167 "aSintactico.y"
        {
            if (currentClass) {
                currentClass->invariantCount++;
                currentContextIsInvariant = 0;
                printf("DEBUG: INV terminado. Invs: %d, Ops colección en INV: %d\n",
                       currentClass->invariantCount, currentClass->invCollectionOpsCount);
            }
        }
#line 1438 "aSintactico.tab.c"
    break;

  case 17: /* constraint_or_definition: DEF optional_constraint_name COLON expression  */
#line 175 "aSintactico.y"
                                                    { printf("DEBUG: DEF procesado\n"); }
#line 1444 "aSintactico.tab.c"
    break;

  case 19: /* optional_constraint_name: ID  */
#line 178 "aSintactico.y"
                                { safeFree((yyvsp[0].sval)); }
#line 1450 "aSintactico.tab.c"
    break;

  case 22: /* operation_context_body: operation_context_body error  */
#line 184 "aSintactico.y"
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
#line 1472 "aSintactico.tab.c"
    break;

  case 23: /* $@4: %empty  */
#line 205 "aSintactico.y"
        {
            if (currentOperation) currentContextIsOperationContext = 1;
            printf("DEBUG: Parseando PRE\n");
        }
#line 1481 "aSintactico.tab.c"
    break;

  case 24: /* condition: PRE optional_constraint_name COLON $@4 expression  */
#line 210 "aSintactico.y"
        {
            if (currentOperation) {
                currentContextIsOperationContext = 0;
                printf("DEBUG: PRE terminado. Ops colección: %d\n", currentOperation->collectionOpsCount);
            }
        }
#line 1492 "aSintactico.tab.c"
    break;

  case 25: /* $@5: %empty  */
#line 217 "aSintactico.y"
        {
            if (currentOperation) currentContextIsOperationContext = 1;
            printf("DEBUG: Parseando POST\n");
        }
#line 1501 "aSintactico.tab.c"
    break;

  case 26: /* condition: POST optional_constraint_name COLON $@5 expression  */
#line 222 "aSintactico.y"
        {
            if (currentOperation) {
                currentContextIsOperationContext = 0;
                printf("DEBUG: POST terminado. Ops colección: %d\n", currentOperation->collectionOpsCount);
            }
        }
#line 1512 "aSintactico.tab.c"
    break;

  case 28: /* expression: expression LOGIC_OP expression  */
#line 232 "aSintactico.y"
                                         { safeFree((yyvsp[-1].sval)); }
#line 1518 "aSintactico.tab.c"
    break;

  case 29: /* expression: expression REL_OP expression  */
#line 233 "aSintactico.y"
                                         { safeFree((yyvsp[-1].sval)); }
#line 1524 "aSintactico.tab.c"
    break;

  case 30: /* expression: expression IMPLIES expression  */
#line 234 "aSintactico.y"
                                         { safeFree((yyvsp[-1].sval)); }
#line 1530 "aSintactico.tab.c"
    break;

  case 35: /* expression: expression ARITH_OP expression  */
#line 239 "aSintactico.y"
                                         { safeFree((yyvsp[-1].sval)); }
#line 1536 "aSintactico.tab.c"
    break;

  case 43: /* expr_primaria: ID  */
#line 250 "aSintactico.y"
         { safeFree((yyvsp[0].sval)); }
#line 1542 "aSintactico.tab.c"
    break;

  case 44: /* expr_primaria: ID DCOLON ID  */
#line 251 "aSintactico.y"
                   { safeFree((yyvsp[-2].sval)); safeFree((yyvsp[0].sval)); }
#line 1548 "aSintactico.tab.c"
    break;

  case 45: /* expr_primaria: expr_primaria DCOLON ID  */
#line 252 "aSintactico.y"
                              { safeFree((yyvsp[0].sval)); }
#line 1554 "aSintactico.tab.c"
    break;

  case 47: /* expr_primaria: AT_SIGN ID  */
#line 254 "aSintactico.y"
                 { safeFree((yyvsp[0].sval)); }
#line 1560 "aSintactico.tab.c"
    break;

  case 51: /* expr_primaria: expr_primaria DOT ID  */
#line 262 "aSintactico.y"
                           { safeFree((yyvsp[0].sval)); }
#line 1566 "aSintactico.tab.c"
    break;

  case 52: /* expr_primaria: expr_primaria DOT ID LPAREN opt_argument_list RPAREN  */
#line 263 "aSintactico.y"
                                                           { safeFree((yyvsp[-3].sval)); }
#line 1572 "aSintactico.tab.c"
    break;

  case 53: /* expr_primaria: expr_primaria ID LPAREN opt_argument_list RPAREN expr_suffix  */
#line 265 "aSintactico.y"
                                                                   { safeFree((yyvsp[-4].sval)); }
#line 1578 "aSintactico.tab.c"
    break;

  case 54: /* expr_primaria: ID LPAREN opt_argument_list RPAREN  */
#line 268 "aSintactico.y"
                                         {
    printf("DEBUG: Función global: %s\n", (yyvsp[-3].sval));
    safeFree((yyvsp[-3].sval));
}
#line 1587 "aSintactico.tab.c"
    break;

  case 55: /* expr_primaria: expr_primaria COLLECTION_OP ID LPAREN opt_param_list BARRA expression RPAREN  */
#line 279 "aSintactico.y"
        {
            if (currentContextIsInvariant && currentClass)
                currentClass->invCollectionOpsCount++;
            else if (currentContextIsOperationContext && currentOperation)
                currentOperation->collectionOpsCount++;
            printf("DEBUG: CollectionOp (con iterador): %s\n", (yyvsp[-6].sval));
            safeFree((yyvsp[-6].sval)); safeFree((yyvsp[-5].sval));
        }
#line 1600 "aSintactico.tab.c"
    break;

  case 56: /* expr_primaria: expr_primaria COLLECTION_OP LPAREN opt_argument_list RPAREN  */
#line 290 "aSintactico.y"
        {
            if (currentContextIsInvariant && currentClass)
                currentClass->invCollectionOpsCount++;
            else if (currentContextIsOperationContext && currentOperation)
                currentOperation->collectionOpsCount++;
            printf("DEBUG: CollectionOp (con argumentos): %s\n", (yyvsp[-3].sval));
            safeFree((yyvsp[-3].sval));
        }
#line 1613 "aSintactico.tab.c"
    break;

  case 57: /* expr_primaria: expr_primaria COLLECTION_OP LPAREN RPAREN  */
#line 301 "aSintactico.y"
        {
            if (currentContextIsInvariant && currentClass)
                currentClass->invCollectionOpsCount++;
            else if (currentContextIsOperationContext && currentOperation)
                currentOperation->collectionOpsCount++;
            printf("DEBUG: CollectionOp (sin args): %s\n", (yyvsp[-2].sval));
            safeFree((yyvsp[-2].sval));
        }
#line 1626 "aSintactico.tab.c"
    break;

  case 58: /* expr_primaria: expr_primaria COLLECTION_OP LPAREN RPAREN REL_OP expression  */
#line 311 "aSintactico.y"
{
    if (currentContextIsInvariant && currentClass)
        currentClass->invCollectionOpsCount++;
    else if (currentContextIsOperationContext && currentOperation)
        currentOperation->collectionOpsCount++;
    printf("DEBUG: CollectionOp + REL_OP: %s\n", (yyvsp[-4].sval));
    safeFree((yyvsp[-4].sval));
}
#line 1639 "aSintactico.tab.c"
    break;

  case 60: /* expr_suffix: DOT ID expr_suffix  */
#line 325 "aSintactico.y"
                         { safeFree((yyvsp[-1].sval)); }
#line 1645 "aSintactico.tab.c"
    break;

  case 61: /* expr_suffix: DOT ID LPAREN opt_argument_list RPAREN expr_suffix  */
#line 326 "aSintactico.y"
                                                         { safeFree((yyvsp[-4].sval)); }
#line 1651 "aSintactico.tab.c"
    break;

  case 66: /* literal: STRING_LITERAL  */
#line 338 "aSintactico.y"
                     { safeFree((yyvsp[0].sval)); }
#line 1657 "aSintactico.tab.c"
    break;

  case 83: /* param_declaration: ID opt_type_specifier  */
#line 358 "aSintactico.y"
                                          { safeFree((yyvsp[-1].sval)); }
#line 1663 "aSintactico.tab.c"
    break;

  case 89: /* type: ID  */
#line 361 "aSintactico.y"
                           { safeFree((yyvsp[0].sval)); }
#line 1669 "aSintactico.tab.c"
    break;


#line 1673 "aSintactico.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 364 "aSintactico.y"


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



