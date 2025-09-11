grammar SimpleClassModel;
import Lexico;

classModelSpecification
    : (classifier | association)*
    ;

classifier
    : classDefinition
    | datatypeDefinition
    | enumeration
    ;

classDefinition
    : 'class' identifier ('extends' identifier)? ('implements' idList)? '{' classBody? '}'
    ;

classBody
    : classBodyElement+
    ;

classBodyElement
    : attributeDefinition
    | operationDefinition
    ;

attributeDefinition
    : 'attribute' identifier ('identity' | 'derived')? ':' type ';'
    | 'static' 'attribute' identifier ':' type ';'
    ;

operationDefinition
    : ('static')? 'operation' identifier '(' parameterDeclarations? ')' ':' type ';'
    ;

parameterDeclarations
    : (parameterDeclaration ',')* parameterDeclaration
    ;

parameterDeclaration
    : identifier ':' type
    ;

idList
    : (identifier ',')* identifier
    ;

datatypeDefinition
    : 'datatype' identifier '=' '{' datatypeBodyElement* '}' 
    ; 

datatypeBodyElement
    : 'field' identifier ':' type ';'
    | operationDefinition
    ;

enumeration
    : 'enumeration' identifier '{' enumerationLiteral+ '}' 
    ;

enumerationLiteral
    : 'literal' identifier ';'
    ;

association
    : 'association' '{' associationName?  associationEndA=associationEnd   associationEndB=associationEnd  associationClass? '}'
    ;

associationClass
    : 'with association class' '{' classBody '}'
    ;

associationName
    : 'name' '=' identifier
    ;

associationEnd
    : 'target' '=' identifier multiplicity? ('<>'|'<<>>')? (('+'|'-')identifier)? constraints?
    ;

multiplicity
    : '*'
    | '0..*'
    | '1..*'
    | '1'
    | '0..1'
    | INT '..' INT
    ;

constraints
    : '{' ('unique' | 'ordered')+ '}'
    ;
 
identifier
    : ID
    ;

type
    : primitiveType
    | identifier
    ;

primitiveType
    : 'Boolean'
    | 'Integer'
    | 'Real'
    | 'String'
;


