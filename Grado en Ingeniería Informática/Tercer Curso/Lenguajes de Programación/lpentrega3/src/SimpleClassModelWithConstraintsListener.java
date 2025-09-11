// Generated from SimpleClassModelWithConstraints.g4 by ANTLR 4.13.2
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link SimpleClassModelWithConstraintsParser}.
 */
public interface SimpleClassModelWithConstraintsListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#model}.
	 * @param ctx the parse tree
	 */
	void enterModel(SimpleClassModelWithConstraintsParser.ModelContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#model}.
	 * @param ctx the parse tree
	 */
	void exitModel(SimpleClassModelWithConstraintsParser.ModelContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classModelSpecification}.
	 * @param ctx the parse tree
	 */
	void enterClassModelSpecification(SimpleClassModelWithConstraintsParser.ClassModelSpecificationContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classModelSpecification}.
	 * @param ctx the parse tree
	 */
	void exitClassModelSpecification(SimpleClassModelWithConstraintsParser.ClassModelSpecificationContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classifier}.
	 * @param ctx the parse tree
	 */
	void enterClassifier(SimpleClassModelWithConstraintsParser.ClassifierContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classifier}.
	 * @param ctx the parse tree
	 */
	void exitClassifier(SimpleClassModelWithConstraintsParser.ClassifierContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classDefinition}.
	 * @param ctx the parse tree
	 */
	void enterClassDefinition(SimpleClassModelWithConstraintsParser.ClassDefinitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classDefinition}.
	 * @param ctx the parse tree
	 */
	void exitClassDefinition(SimpleClassModelWithConstraintsParser.ClassDefinitionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classBody}.
	 * @param ctx the parse tree
	 */
	void enterClassBody(SimpleClassModelWithConstraintsParser.ClassBodyContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classBody}.
	 * @param ctx the parse tree
	 */
	void exitClassBody(SimpleClassModelWithConstraintsParser.ClassBodyContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classBodyElement}.
	 * @param ctx the parse tree
	 */
	void enterClassBodyElement(SimpleClassModelWithConstraintsParser.ClassBodyElementContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classBodyElement}.
	 * @param ctx the parse tree
	 */
	void exitClassBodyElement(SimpleClassModelWithConstraintsParser.ClassBodyElementContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#attributeDefinition}.
	 * @param ctx the parse tree
	 */
	void enterAttributeDefinition(SimpleClassModelWithConstraintsParser.AttributeDefinitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#attributeDefinition}.
	 * @param ctx the parse tree
	 */
	void exitAttributeDefinition(SimpleClassModelWithConstraintsParser.AttributeDefinitionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#operationDefinition}.
	 * @param ctx the parse tree
	 */
	void enterOperationDefinition(SimpleClassModelWithConstraintsParser.OperationDefinitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#operationDefinition}.
	 * @param ctx the parse tree
	 */
	void exitOperationDefinition(SimpleClassModelWithConstraintsParser.OperationDefinitionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#parameterDeclarations}.
	 * @param ctx the parse tree
	 */
	void enterParameterDeclarations(SimpleClassModelWithConstraintsParser.ParameterDeclarationsContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#parameterDeclarations}.
	 * @param ctx the parse tree
	 */
	void exitParameterDeclarations(SimpleClassModelWithConstraintsParser.ParameterDeclarationsContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#parameterDeclaration}.
	 * @param ctx the parse tree
	 */
	void enterParameterDeclaration(SimpleClassModelWithConstraintsParser.ParameterDeclarationContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#parameterDeclaration}.
	 * @param ctx the parse tree
	 */
	void exitParameterDeclaration(SimpleClassModelWithConstraintsParser.ParameterDeclarationContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#idList}.
	 * @param ctx the parse tree
	 */
	void enterIdList(SimpleClassModelWithConstraintsParser.IdListContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#idList}.
	 * @param ctx the parse tree
	 */
	void exitIdList(SimpleClassModelWithConstraintsParser.IdListContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#datatypeDefinition}.
	 * @param ctx the parse tree
	 */
	void enterDatatypeDefinition(SimpleClassModelWithConstraintsParser.DatatypeDefinitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#datatypeDefinition}.
	 * @param ctx the parse tree
	 */
	void exitDatatypeDefinition(SimpleClassModelWithConstraintsParser.DatatypeDefinitionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#datatypeBodyElement}.
	 * @param ctx the parse tree
	 */
	void enterDatatypeBodyElement(SimpleClassModelWithConstraintsParser.DatatypeBodyElementContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#datatypeBodyElement}.
	 * @param ctx the parse tree
	 */
	void exitDatatypeBodyElement(SimpleClassModelWithConstraintsParser.DatatypeBodyElementContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#enumeration}.
	 * @param ctx the parse tree
	 */
	void enterEnumeration(SimpleClassModelWithConstraintsParser.EnumerationContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#enumeration}.
	 * @param ctx the parse tree
	 */
	void exitEnumeration(SimpleClassModelWithConstraintsParser.EnumerationContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#enumerationLiteral}.
	 * @param ctx the parse tree
	 */
	void enterEnumerationLiteral(SimpleClassModelWithConstraintsParser.EnumerationLiteralContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#enumerationLiteral}.
	 * @param ctx the parse tree
	 */
	void exitEnumerationLiteral(SimpleClassModelWithConstraintsParser.EnumerationLiteralContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#association}.
	 * @param ctx the parse tree
	 */
	void enterAssociation(SimpleClassModelWithConstraintsParser.AssociationContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#association}.
	 * @param ctx the parse tree
	 */
	void exitAssociation(SimpleClassModelWithConstraintsParser.AssociationContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationClass}.
	 * @param ctx the parse tree
	 */
	void enterAssociationClass(SimpleClassModelWithConstraintsParser.AssociationClassContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationClass}.
	 * @param ctx the parse tree
	 */
	void exitAssociationClass(SimpleClassModelWithConstraintsParser.AssociationClassContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationName}.
	 * @param ctx the parse tree
	 */
	void enterAssociationName(SimpleClassModelWithConstraintsParser.AssociationNameContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationName}.
	 * @param ctx the parse tree
	 */
	void exitAssociationName(SimpleClassModelWithConstraintsParser.AssociationNameContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationEnd}.
	 * @param ctx the parse tree
	 */
	void enterAssociationEnd(SimpleClassModelWithConstraintsParser.AssociationEndContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationEnd}.
	 * @param ctx the parse tree
	 */
	void exitAssociationEnd(SimpleClassModelWithConstraintsParser.AssociationEndContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#multiplicity}.
	 * @param ctx the parse tree
	 */
	void enterMultiplicity(SimpleClassModelWithConstraintsParser.MultiplicityContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#multiplicity}.
	 * @param ctx the parse tree
	 */
	void exitMultiplicity(SimpleClassModelWithConstraintsParser.MultiplicityContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#constraints}.
	 * @param ctx the parse tree
	 */
	void enterConstraints(SimpleClassModelWithConstraintsParser.ConstraintsContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#constraints}.
	 * @param ctx the parse tree
	 */
	void exitConstraints(SimpleClassModelWithConstraintsParser.ConstraintsContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#identifier}.
	 * @param ctx the parse tree
	 */
	void enterIdentifier(SimpleClassModelWithConstraintsParser.IdentifierContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#identifier}.
	 * @param ctx the parse tree
	 */
	void exitIdentifier(SimpleClassModelWithConstraintsParser.IdentifierContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#type}.
	 * @param ctx the parse tree
	 */
	void enterType(SimpleClassModelWithConstraintsParser.TypeContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#type}.
	 * @param ctx the parse tree
	 */
	void exitType(SimpleClassModelWithConstraintsParser.TypeContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#primitiveType}.
	 * @param ctx the parse tree
	 */
	void enterPrimitiveType(SimpleClassModelWithConstraintsParser.PrimitiveTypeContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#primitiveType}.
	 * @param ctx the parse tree
	 */
	void exitPrimitiveType(SimpleClassModelWithConstraintsParser.PrimitiveTypeContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#multipleContextSpecifications}.
	 * @param ctx the parse tree
	 */
	void enterMultipleContextSpecifications(SimpleClassModelWithConstraintsParser.MultipleContextSpecificationsContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#multipleContextSpecifications}.
	 * @param ctx the parse tree
	 */
	void exitMultipleContextSpecifications(SimpleClassModelWithConstraintsParser.MultipleContextSpecificationsContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#invariantContext}.
	 * @param ctx the parse tree
	 */
	void enterInvariantContext(SimpleClassModelWithConstraintsParser.InvariantContextContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#invariantContext}.
	 * @param ctx the parse tree
	 */
	void exitInvariantContext(SimpleClassModelWithConstraintsParser.InvariantContextContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#invariant}.
	 * @param ctx the parse tree
	 */
	void enterInvariant(SimpleClassModelWithConstraintsParser.InvariantContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#invariant}.
	 * @param ctx the parse tree
	 */
	void exitInvariant(SimpleClassModelWithConstraintsParser.InvariantContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#prepostContext}.
	 * @param ctx the parse tree
	 */
	void enterPrepostContext(SimpleClassModelWithConstraintsParser.PrepostContextContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#prepostContext}.
	 * @param ctx the parse tree
	 */
	void exitPrepostContext(SimpleClassModelWithConstraintsParser.PrepostContextContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#operationDeclaration}.
	 * @param ctx the parse tree
	 */
	void enterOperationDeclaration(SimpleClassModelWithConstraintsParser.OperationDeclarationContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#operationDeclaration}.
	 * @param ctx the parse tree
	 */
	void exitOperationDeclaration(SimpleClassModelWithConstraintsParser.OperationDeclarationContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#prepostSpecification}.
	 * @param ctx the parse tree
	 */
	void enterPrepostSpecification(SimpleClassModelWithConstraintsParser.PrepostSpecificationContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#prepostSpecification}.
	 * @param ctx the parse tree
	 */
	void exitPrepostSpecification(SimpleClassModelWithConstraintsParser.PrepostSpecificationContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#precondition}.
	 * @param ctx the parse tree
	 */
	void enterPrecondition(SimpleClassModelWithConstraintsParser.PreconditionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#precondition}.
	 * @param ctx the parse tree
	 */
	void exitPrecondition(SimpleClassModelWithConstraintsParser.PreconditionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#postcondition}.
	 * @param ctx the parse tree
	 */
	void enterPostcondition(SimpleClassModelWithConstraintsParser.PostconditionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#postcondition}.
	 * @param ctx the parse tree
	 */
	void exitPostcondition(SimpleClassModelWithConstraintsParser.PostconditionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#ocltype}.
	 * @param ctx the parse tree
	 */
	void enterOcltype(SimpleClassModelWithConstraintsParser.OcltypeContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#ocltype}.
	 * @param ctx the parse tree
	 */
	void exitOcltype(SimpleClassModelWithConstraintsParser.OcltypeContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#expressionList}.
	 * @param ctx the parse tree
	 */
	void enterExpressionList(SimpleClassModelWithConstraintsParser.ExpressionListContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#expressionList}.
	 * @param ctx the parse tree
	 */
	void exitExpressionList(SimpleClassModelWithConstraintsParser.ExpressionListContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterExpression(SimpleClassModelWithConstraintsParser.ExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitExpression(SimpleClassModelWithConstraintsParser.ExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#basicExpression}.
	 * @param ctx the parse tree
	 */
	void enterBasicExpression(SimpleClassModelWithConstraintsParser.BasicExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#basicExpression}.
	 * @param ctx the parse tree
	 */
	void exitBasicExpression(SimpleClassModelWithConstraintsParser.BasicExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#conditionalExpression}.
	 * @param ctx the parse tree
	 */
	void enterConditionalExpression(SimpleClassModelWithConstraintsParser.ConditionalExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#conditionalExpression}.
	 * @param ctx the parse tree
	 */
	void exitConditionalExpression(SimpleClassModelWithConstraintsParser.ConditionalExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#lambdaExpression}.
	 * @param ctx the parse tree
	 */
	void enterLambdaExpression(SimpleClassModelWithConstraintsParser.LambdaExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#lambdaExpression}.
	 * @param ctx the parse tree
	 */
	void exitLambdaExpression(SimpleClassModelWithConstraintsParser.LambdaExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#letExpression}.
	 * @param ctx the parse tree
	 */
	void enterLetExpression(SimpleClassModelWithConstraintsParser.LetExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#letExpression}.
	 * @param ctx the parse tree
	 */
	void exitLetExpression(SimpleClassModelWithConstraintsParser.LetExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#logicalExpression}.
	 * @param ctx the parse tree
	 */
	void enterLogicalExpression(SimpleClassModelWithConstraintsParser.LogicalExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#logicalExpression}.
	 * @param ctx the parse tree
	 */
	void exitLogicalExpression(SimpleClassModelWithConstraintsParser.LogicalExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#equalityExpression}.
	 * @param ctx the parse tree
	 */
	void enterEqualityExpression(SimpleClassModelWithConstraintsParser.EqualityExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#equalityExpression}.
	 * @param ctx the parse tree
	 */
	void exitEqualityExpression(SimpleClassModelWithConstraintsParser.EqualityExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#additiveExpression}.
	 * @param ctx the parse tree
	 */
	void enterAdditiveExpression(SimpleClassModelWithConstraintsParser.AdditiveExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#additiveExpression}.
	 * @param ctx the parse tree
	 */
	void exitAdditiveExpression(SimpleClassModelWithConstraintsParser.AdditiveExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#factorExpression}.
	 * @param ctx the parse tree
	 */
	void enterFactorExpression(SimpleClassModelWithConstraintsParser.FactorExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#factorExpression}.
	 * @param ctx the parse tree
	 */
	void exitFactorExpression(SimpleClassModelWithConstraintsParser.FactorExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#factor2Expression}.
	 * @param ctx the parse tree
	 */
	void enterFactor2Expression(SimpleClassModelWithConstraintsParser.Factor2ExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#factor2Expression}.
	 * @param ctx the parse tree
	 */
	void exitFactor2Expression(SimpleClassModelWithConstraintsParser.Factor2ExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#identOptType}.
	 * @param ctx the parse tree
	 */
	void enterIdentOptType(SimpleClassModelWithConstraintsParser.IdentOptTypeContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#identOptType}.
	 * @param ctx the parse tree
	 */
	void exitIdentOptType(SimpleClassModelWithConstraintsParser.IdentOptTypeContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#setExpression}.
	 * @param ctx the parse tree
	 */
	void enterSetExpression(SimpleClassModelWithConstraintsParser.SetExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#setExpression}.
	 * @param ctx the parse tree
	 */
	void exitSetExpression(SimpleClassModelWithConstraintsParser.SetExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link SimpleClassModelWithConstraintsParser#qualified_name}.
	 * @param ctx the parse tree
	 */
	void enterQualified_name(SimpleClassModelWithConstraintsParser.Qualified_nameContext ctx);
	/**
	 * Exit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#qualified_name}.
	 * @param ctx the parse tree
	 */
	void exitQualified_name(SimpleClassModelWithConstraintsParser.Qualified_nameContext ctx);
}