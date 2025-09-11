// Generated from SimpleClassModelWithConstraints.g4 by ANTLR 4.13.2
import org.antlr.v4.runtime.tree.ParseTreeVisitor;

/**
 * This interface defines a complete generic visitor for a parse tree produced
 * by {@link SimpleClassModelWithConstraintsParser}.
 *
 * @param <T> The return type of the visit operation. Use {@link Void} for
 * operations with no return type.
 */
public interface SimpleClassModelWithConstraintsVisitor<T> extends ParseTreeVisitor<T> {
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#model}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitModel(SimpleClassModelWithConstraintsParser.ModelContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classModelSpecification}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitClassModelSpecification(SimpleClassModelWithConstraintsParser.ClassModelSpecificationContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classifier}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitClassifier(SimpleClassModelWithConstraintsParser.ClassifierContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classDefinition}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitClassDefinition(SimpleClassModelWithConstraintsParser.ClassDefinitionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classBody}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitClassBody(SimpleClassModelWithConstraintsParser.ClassBodyContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#classBodyElement}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitClassBodyElement(SimpleClassModelWithConstraintsParser.ClassBodyElementContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#attributeDefinition}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitAttributeDefinition(SimpleClassModelWithConstraintsParser.AttributeDefinitionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#operationDefinition}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitOperationDefinition(SimpleClassModelWithConstraintsParser.OperationDefinitionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#parameterDeclarations}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitParameterDeclarations(SimpleClassModelWithConstraintsParser.ParameterDeclarationsContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#parameterDeclaration}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitParameterDeclaration(SimpleClassModelWithConstraintsParser.ParameterDeclarationContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#idList}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitIdList(SimpleClassModelWithConstraintsParser.IdListContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#datatypeDefinition}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitDatatypeDefinition(SimpleClassModelWithConstraintsParser.DatatypeDefinitionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#datatypeBodyElement}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitDatatypeBodyElement(SimpleClassModelWithConstraintsParser.DatatypeBodyElementContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#enumeration}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitEnumeration(SimpleClassModelWithConstraintsParser.EnumerationContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#enumerationLiteral}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitEnumerationLiteral(SimpleClassModelWithConstraintsParser.EnumerationLiteralContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#association}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitAssociation(SimpleClassModelWithConstraintsParser.AssociationContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationClass}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitAssociationClass(SimpleClassModelWithConstraintsParser.AssociationClassContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationName}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitAssociationName(SimpleClassModelWithConstraintsParser.AssociationNameContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#associationEnd}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitAssociationEnd(SimpleClassModelWithConstraintsParser.AssociationEndContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#multiplicity}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitMultiplicity(SimpleClassModelWithConstraintsParser.MultiplicityContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#constraints}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitConstraints(SimpleClassModelWithConstraintsParser.ConstraintsContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#identifier}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitIdentifier(SimpleClassModelWithConstraintsParser.IdentifierContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#type}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitType(SimpleClassModelWithConstraintsParser.TypeContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#primitiveType}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitPrimitiveType(SimpleClassModelWithConstraintsParser.PrimitiveTypeContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#multipleContextSpecifications}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitMultipleContextSpecifications(SimpleClassModelWithConstraintsParser.MultipleContextSpecificationsContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#invariantContext}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitInvariantContext(SimpleClassModelWithConstraintsParser.InvariantContextContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#invariant}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitInvariant(SimpleClassModelWithConstraintsParser.InvariantContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#prepostContext}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitPrepostContext(SimpleClassModelWithConstraintsParser.PrepostContextContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#operationDeclaration}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitOperationDeclaration(SimpleClassModelWithConstraintsParser.OperationDeclarationContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#prepostSpecification}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitPrepostSpecification(SimpleClassModelWithConstraintsParser.PrepostSpecificationContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#precondition}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitPrecondition(SimpleClassModelWithConstraintsParser.PreconditionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#postcondition}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitPostcondition(SimpleClassModelWithConstraintsParser.PostconditionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#ocltype}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitOcltype(SimpleClassModelWithConstraintsParser.OcltypeContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#expressionList}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitExpressionList(SimpleClassModelWithConstraintsParser.ExpressionListContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#expression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitExpression(SimpleClassModelWithConstraintsParser.ExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#basicExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitBasicExpression(SimpleClassModelWithConstraintsParser.BasicExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#conditionalExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitConditionalExpression(SimpleClassModelWithConstraintsParser.ConditionalExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#lambdaExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitLambdaExpression(SimpleClassModelWithConstraintsParser.LambdaExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#letExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitLetExpression(SimpleClassModelWithConstraintsParser.LetExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#logicalExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitLogicalExpression(SimpleClassModelWithConstraintsParser.LogicalExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#equalityExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitEqualityExpression(SimpleClassModelWithConstraintsParser.EqualityExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#additiveExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitAdditiveExpression(SimpleClassModelWithConstraintsParser.AdditiveExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#factorExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitFactorExpression(SimpleClassModelWithConstraintsParser.FactorExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#factor2Expression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitFactor2Expression(SimpleClassModelWithConstraintsParser.Factor2ExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#identOptType}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitIdentOptType(SimpleClassModelWithConstraintsParser.IdentOptTypeContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#setExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitSetExpression(SimpleClassModelWithConstraintsParser.SetExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link SimpleClassModelWithConstraintsParser#qualified_name}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitQualified_name(SimpleClassModelWithConstraintsParser.Qualified_nameContext ctx);
}