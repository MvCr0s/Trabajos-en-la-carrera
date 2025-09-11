grammar Lexico;

FLOAT_LITERAL
    : Digits '.' Digits
    ;

STRING1_LITERAL
    : '"' (~["\\\r\n] | EscapeSequence)* '"'
    ;

STRING2_LITERAL
    : '\'' (~['\\\r\n] | EscapeSequence)* '\'';

ENUMERATION_LITERAL
    : ID '::' ID
    ;

NULL_LITERAL
    : 'null'
    ;

MULTILINE_COMMENT
    : '/*' .*? '*/' -> channel(HIDDEN)
    ;

fragment EscapeSequence
    : '\\' [btnfr"'\\]
    | '\\' ([0-3]? [0-7])? [0-7]
    | '\\' 'u' HexDigit HexDigit HexDigit HexDigit
    ;

fragment HexDigits
    : HexDigit ((HexDigit | '_')* HexDigit)?
    ;

fragment HexDigit
    : [0-9a-fA-F]
    ;

fragment Digits
    : [0-9]+
    ;

NEWLINE
    : [\r\n]+ -> skip
    ;

INT
    : [0-9]+
    ;

ID
    : [a-zA-Z_$]+ [a-zA-Z0-9_$]*
    ; // match identifiers

WS
    : [ \t\n\r]+ -> skip
    ;
