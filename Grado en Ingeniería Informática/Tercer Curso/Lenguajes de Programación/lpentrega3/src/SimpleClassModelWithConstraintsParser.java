// Generated from SimpleClassModelWithConstraints.g4 by ANTLR 4.13.2
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue", "this-escape"})
public class SimpleClassModelWithConstraintsParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, T__6=7, T__7=8, T__8=9, 
		T__9=10, T__10=11, T__11=12, T__12=13, T__13=14, T__14=15, T__15=16, T__16=17, 
		T__17=18, T__18=19, T__19=20, T__20=21, T__21=22, T__22=23, T__23=24, 
		T__24=25, T__25=26, T__26=27, T__27=28, T__28=29, T__29=30, T__30=31, 
		T__31=32, T__32=33, T__33=34, T__34=35, T__35=36, T__36=37, T__37=38, 
		T__38=39, T__39=40, T__40=41, T__41=42, T__42=43, T__43=44, T__44=45, 
		T__45=46, T__46=47, T__47=48, T__48=49, T__49=50, T__50=51, T__51=52, 
		T__52=53, T__53=54, T__54=55, T__55=56, T__56=57, T__57=58, T__58=59, 
		T__59=60, T__60=61, T__61=62, T__62=63, T__63=64, T__64=65, T__65=66, 
		T__66=67, T__67=68, T__68=69, T__69=70, T__70=71, T__71=72, T__72=73, 
		T__73=74, T__74=75, T__75=76, T__76=77, T__77=78, T__78=79, T__79=80, 
		T__80=81, T__81=82, T__82=83, T__83=84, T__84=85, T__85=86, T__86=87, 
		T__87=88, T__88=89, T__89=90, T__90=91, T__91=92, T__92=93, T__93=94, 
		T__94=95, T__95=96, T__96=97, T__97=98, T__98=99, T__99=100, T__100=101, 
		T__101=102, T__102=103, T__103=104, T__104=105, T__105=106, T__106=107, 
		T__107=108, T__108=109, T__109=110, T__110=111, T__111=112, T__112=113, 
		T__113=114, T__114=115, T__115=116, T__116=117, T__117=118, T__118=119, 
		T__119=120, T__120=121, T__121=122, T__122=123, T__123=124, T__124=125, 
		T__125=126, T__126=127, T__127=128, T__128=129, T__129=130, T__130=131, 
		T__131=132, T__132=133, T__133=134, T__134=135, T__135=136, T__136=137, 
		T__137=138, T__138=139, T__139=140, T__140=141, T__141=142, T__142=143, 
		T__143=144, T__144=145, T__145=146, T__146=147, T__147=148, T__148=149, 
		T__149=150, T__150=151, T__151=152, T__152=153, T__153=154, T__154=155, 
		T__155=156, T__156=157, T__157=158, T__158=159, T__159=160, T__160=161, 
		T__161=162, T__162=163, T__163=164, T__164=165, T__165=166, T__166=167, 
		T__167=168, T__168=169, T__169=170, T__170=171, T__171=172, T__172=173, 
		T__173=174, T__174=175, T__175=176, T__176=177, T__177=178, T__178=179, 
		T__179=180, T__180=181, T__181=182, T__182=183, T__183=184, T__184=185, 
		T__185=186, T__186=187, T__187=188, T__188=189, INT=190, FLOAT_LITERAL=191, 
		STRING1_LITERAL=192, STRING2_LITERAL=193, ENUMERATION_LITERAL=194, NULL_LITERAL=195, 
		ID=196, MULTILINE_COMMENT=197, NEWLINE=198, WS=199;
	public static final int
		RULE_model = 0, RULE_classModelSpecification = 1, RULE_classifier = 2, 
		RULE_classDefinition = 3, RULE_classBody = 4, RULE_classBodyElement = 5, 
		RULE_attributeDefinition = 6, RULE_operationDefinition = 7, RULE_parameterDeclarations = 8, 
		RULE_parameterDeclaration = 9, RULE_idList = 10, RULE_datatypeDefinition = 11, 
		RULE_datatypeBodyElement = 12, RULE_enumeration = 13, RULE_enumerationLiteral = 14, 
		RULE_association = 15, RULE_associationClass = 16, RULE_associationName = 17, 
		RULE_associationEnd = 18, RULE_multiplicity = 19, RULE_constraints = 20, 
		RULE_identifier = 21, RULE_type = 22, RULE_primitiveType = 23, RULE_multipleContextSpecifications = 24, 
		RULE_invariantContext = 25, RULE_invariant = 26, RULE_prepostContext = 27, 
		RULE_operationDeclaration = 28, RULE_prepostSpecification = 29, RULE_precondition = 30, 
		RULE_postcondition = 31, RULE_ocltype = 32, RULE_expressionList = 33, 
		RULE_expression = 34, RULE_basicExpression = 35, RULE_conditionalExpression = 36, 
		RULE_lambdaExpression = 37, RULE_letExpression = 38, RULE_logicalExpression = 39, 
		RULE_equalityExpression = 40, RULE_additiveExpression = 41, RULE_factorExpression = 42, 
		RULE_factor2Expression = 43, RULE_identOptType = 44, RULE_setExpression = 45, 
		RULE_qualified_name = 46;
	private static String[] makeRuleNames() {
		return new String[] {
			"model", "classModelSpecification", "classifier", "classDefinition", 
			"classBody", "classBodyElement", "attributeDefinition", "operationDefinition", 
			"parameterDeclarations", "parameterDeclaration", "idList", "datatypeDefinition", 
			"datatypeBodyElement", "enumeration", "enumerationLiteral", "association", 
			"associationClass", "associationName", "associationEnd", "multiplicity", 
			"constraints", "identifier", "type", "primitiveType", "multipleContextSpecifications", 
			"invariantContext", "invariant", "prepostContext", "operationDeclaration", 
			"prepostSpecification", "precondition", "postcondition", "ocltype", "expressionList", 
			"expression", "basicExpression", "conditionalExpression", "lambdaExpression", 
			"letExpression", "logicalExpression", "equalityExpression", "additiveExpression", 
			"factorExpression", "factor2Expression", "identOptType", "setExpression", 
			"qualified_name"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'class'", "'extends'", "'implements'", "'{'", "'}'", "'attribute'", 
			"'identity'", "'derived'", "':'", "';'", "'static'", "'operation'", "'('", 
			"')'", "','", "'datatype'", "'='", "'field'", "'enumeration'", "'literal'", 
			"'association'", "'with association class'", "'name'", "'target'", "'<>'", 
			"'<<>>'", "'+'", "'-'", "'*'", "'0..*'", "'1..*'", "'..'", "'unique'", 
			"'ordered'", "'Boolean'", "'Integer'", "'Real'", "'String'", "'context'", 
			"'inv'", "'::'", "'pre:'", "'post:'", "'Sequence'", "'Set'", "'Bag'", 
			"'OrderedSet'", "'Ref'", "'Map'", "'Function'", "'.'", "'['", "']'", 
			"'@pre'", "'if'", "'then'", "'else'", "'endif'", "'lambda'", "'in'", 
			"'let'", "'not'", "'and'", "'&'", "'or'", "'xor'", "'=>'", "'implies'", 
			"'<'", "'>'", "'>='", "'<='", "'/='", "'/:'", "'<:'", "'|->'", "'/'", 
			"'mod'", "'div'", "'?'", "'!'", "'->size()'", "'->copy()'", "'->isEmpty()'", 
			"'->notEmpty()'", "'->asSet()'", "'->asBag()'", "'->asOrderedSet()'", 
			"'->asSequence()'", "'->sort()'", "'->any()'", "'->log()'", "'->exp()'", 
			"'->sin()'", "'->cos()'", "'->tan()'", "'->asin()'", "'->acos()'", "'->atan()'", 
			"'->log10()'", "'->first()'", "'->last()'", "'->front()'", "'->tail()'", 
			"'->reverse()'", "'->tanh()'", "'->sinh()'", "'->cosh()'", "'->floor()'", 
			"'->ceil()'", "'->round()'", "'->abs()'", "'->oclType()'", "'->allInstances()'", 
			"'->oclIsUndefined()'", "'->oclIsInvalid()'", "'->oclIsNew()'", "'->sum()'", 
			"'->prd()'", "'->max()'", "'->min()'", "'->sqrt()'", "'->cbrt()'", "'->sqr()'", 
			"'->characters()'", "'->toInteger()'", "'->toReal()'", "'->toBoolean()'", 
			"'->display()'", "'->toUpperCase()'", "'->toLowerCase()'", "'->unionAll()'", 
			"'->intersectAll()'", "'->concatenateAll()'", "'->pow'", "'->gcd'", "'->at'", 
			"'->union'", "'->intersection'", "'->includes'", "'->excludes'", "'->including'", 
			"'->excluding'", "'->includesAll'", "'->symmetricDifference'", "'->excludesAll'", 
			"'->prepend'", "'->append'", "'->count'", "'->apply'", "'->hasMatch'", 
			"'->isMatch'", "'->firstMatch'", "'->indexOf'", "'->lastIndexOf'", "'->split'", 
			"'->hasPrefix'", "'->hasSuffix'", "'->equalsIgnoreCase'", "'->oclAsType'", 
			"'->oclIsTypeOf'", "'->oclIsKindOf'", "'->oclAsSet'", "'->collect'", 
			"'|'", "'->select'", "'->reject'", "'->forAll'", "'->exists'", "'->exists1'", 
			"'->one'", "'->any'", "'->closure'", "'->sortedBy'", "'->isUnique'", 
			"'->subrange'", "'->replace'", "'->replaceAll'", "'->replaceAllMatches'", 
			"'->replaceFirstMatch'", "'->insertAt'", "'->insertInto'", "'->setAt'", 
			"'->iterate'", "'OrderedSet{'", "'Bag{'", "'Set{'", "'Sequence{'", "'Map{'", 
			null, null, null, null, null, "'null'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, "INT", "FLOAT_LITERAL", 
			"STRING1_LITERAL", "STRING2_LITERAL", "ENUMERATION_LITERAL", "NULL_LITERAL", 
			"ID", "MULTILINE_COMMENT", "NEWLINE", "WS"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "SimpleClassModelWithConstraints.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public SimpleClassModelWithConstraintsParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ModelContext extends ParserRuleContext {
		public ClassModelSpecificationContext classModelSpecification() {
			return getRuleContext(ClassModelSpecificationContext.class,0);
		}
		public MultipleContextSpecificationsContext multipleContextSpecifications() {
			return getRuleContext(MultipleContextSpecificationsContext.class,0);
		}
		public ModelContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_model; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterModel(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitModel(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitModel(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ModelContext model() throws RecognitionException {
		ModelContext _localctx = new ModelContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_model);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(94);
			classModelSpecification();
			setState(95);
			multipleContextSpecifications();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ClassModelSpecificationContext extends ParserRuleContext {
		public List<ClassifierContext> classifier() {
			return getRuleContexts(ClassifierContext.class);
		}
		public ClassifierContext classifier(int i) {
			return getRuleContext(ClassifierContext.class,i);
		}
		public List<AssociationContext> association() {
			return getRuleContexts(AssociationContext.class);
		}
		public AssociationContext association(int i) {
			return getRuleContext(AssociationContext.class,i);
		}
		public ClassModelSpecificationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_classModelSpecification; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterClassModelSpecification(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitClassModelSpecification(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitClassModelSpecification(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ClassModelSpecificationContext classModelSpecification() throws RecognitionException {
		ClassModelSpecificationContext _localctx = new ClassModelSpecificationContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_classModelSpecification);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(101);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 2686978L) != 0)) {
				{
				setState(99);
				_errHandler.sync(this);
				switch (_input.LA(1)) {
				case T__0:
				case T__15:
				case T__18:
					{
					setState(97);
					classifier();
					}
					break;
				case T__20:
					{
					setState(98);
					association();
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				}
				setState(103);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ClassifierContext extends ParserRuleContext {
		public ClassDefinitionContext classDefinition() {
			return getRuleContext(ClassDefinitionContext.class,0);
		}
		public DatatypeDefinitionContext datatypeDefinition() {
			return getRuleContext(DatatypeDefinitionContext.class,0);
		}
		public EnumerationContext enumeration() {
			return getRuleContext(EnumerationContext.class,0);
		}
		public ClassifierContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_classifier; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterClassifier(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitClassifier(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitClassifier(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ClassifierContext classifier() throws RecognitionException {
		ClassifierContext _localctx = new ClassifierContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_classifier);
		try {
			setState(107);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__0:
				enterOuterAlt(_localctx, 1);
				{
				setState(104);
				classDefinition();
				}
				break;
			case T__15:
				enterOuterAlt(_localctx, 2);
				{
				setState(105);
				datatypeDefinition();
				}
				break;
			case T__18:
				enterOuterAlt(_localctx, 3);
				{
				setState(106);
				enumeration();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ClassDefinitionContext extends ParserRuleContext {
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public IdListContext idList() {
			return getRuleContext(IdListContext.class,0);
		}
		public ClassBodyContext classBody() {
			return getRuleContext(ClassBodyContext.class,0);
		}
		public ClassDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_classDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterClassDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitClassDefinition(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitClassDefinition(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ClassDefinitionContext classDefinition() throws RecognitionException {
		ClassDefinitionContext _localctx = new ClassDefinitionContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_classDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(109);
			match(T__0);
			setState(110);
			identifier();
			setState(113);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__1) {
				{
				setState(111);
				match(T__1);
				setState(112);
				identifier();
				}
			}

			setState(117);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__2) {
				{
				setState(115);
				match(T__2);
				setState(116);
				idList();
				}
			}

			setState(119);
			match(T__3);
			setState(121);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 6208L) != 0)) {
				{
				setState(120);
				classBody();
				}
			}

			setState(123);
			match(T__4);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ClassBodyContext extends ParserRuleContext {
		public List<ClassBodyElementContext> classBodyElement() {
			return getRuleContexts(ClassBodyElementContext.class);
		}
		public ClassBodyElementContext classBodyElement(int i) {
			return getRuleContext(ClassBodyElementContext.class,i);
		}
		public ClassBodyContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_classBody; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterClassBody(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitClassBody(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitClassBody(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ClassBodyContext classBody() throws RecognitionException {
		ClassBodyContext _localctx = new ClassBodyContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_classBody);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(126); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(125);
				classBodyElement();
				}
				}
				setState(128); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( (((_la) & ~0x3f) == 0 && ((1L << _la) & 6208L) != 0) );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ClassBodyElementContext extends ParserRuleContext {
		public AttributeDefinitionContext attributeDefinition() {
			return getRuleContext(AttributeDefinitionContext.class,0);
		}
		public OperationDefinitionContext operationDefinition() {
			return getRuleContext(OperationDefinitionContext.class,0);
		}
		public ClassBodyElementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_classBodyElement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterClassBodyElement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitClassBodyElement(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitClassBodyElement(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ClassBodyElementContext classBodyElement() throws RecognitionException {
		ClassBodyElementContext _localctx = new ClassBodyElementContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_classBodyElement);
		try {
			setState(132);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,7,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(130);
				attributeDefinition();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(131);
				operationDefinition();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AttributeDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public AttributeDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attributeDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterAttributeDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitAttributeDefinition(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitAttributeDefinition(this);
			else return visitor.visitChildren(this);
		}
	}

	public final AttributeDefinitionContext attributeDefinition() throws RecognitionException {
		AttributeDefinitionContext _localctx = new AttributeDefinitionContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_attributeDefinition);
		int _la;
		try {
			setState(150);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__5:
				enterOuterAlt(_localctx, 1);
				{
				setState(134);
				match(T__5);
				setState(135);
				identifier();
				setState(137);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__6 || _la==T__7) {
					{
					setState(136);
					_la = _input.LA(1);
					if ( !(_la==T__6 || _la==T__7) ) {
					_errHandler.recoverInline(this);
					}
					else {
						if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
						_errHandler.reportMatch(this);
						consume();
					}
					}
				}

				setState(139);
				match(T__8);
				setState(140);
				type();
				setState(141);
				match(T__9);
				}
				break;
			case T__10:
				enterOuterAlt(_localctx, 2);
				{
				setState(143);
				match(T__10);
				setState(144);
				match(T__5);
				setState(145);
				identifier();
				setState(146);
				match(T__8);
				setState(147);
				type();
				setState(148);
				match(T__9);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class OperationDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public ParameterDeclarationsContext parameterDeclarations() {
			return getRuleContext(ParameterDeclarationsContext.class,0);
		}
		public OperationDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_operationDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterOperationDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitOperationDefinition(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitOperationDefinition(this);
			else return visitor.visitChildren(this);
		}
	}

	public final OperationDefinitionContext operationDefinition() throws RecognitionException {
		OperationDefinitionContext _localctx = new OperationDefinitionContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_operationDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(153);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__10) {
				{
				setState(152);
				match(T__10);
				}
			}

			setState(155);
			match(T__11);
			setState(156);
			identifier();
			setState(157);
			match(T__12);
			setState(159);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==ID) {
				{
				setState(158);
				parameterDeclarations();
				}
			}

			setState(161);
			match(T__13);
			setState(162);
			match(T__8);
			setState(163);
			type();
			setState(164);
			match(T__9);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ParameterDeclarationsContext extends ParserRuleContext {
		public List<ParameterDeclarationContext> parameterDeclaration() {
			return getRuleContexts(ParameterDeclarationContext.class);
		}
		public ParameterDeclarationContext parameterDeclaration(int i) {
			return getRuleContext(ParameterDeclarationContext.class,i);
		}
		public ParameterDeclarationsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_parameterDeclarations; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterParameterDeclarations(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitParameterDeclarations(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitParameterDeclarations(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ParameterDeclarationsContext parameterDeclarations() throws RecognitionException {
		ParameterDeclarationsContext _localctx = new ParameterDeclarationsContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_parameterDeclarations);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(171);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,12,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(166);
					parameterDeclaration();
					setState(167);
					match(T__14);
					}
					} 
				}
				setState(173);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,12,_ctx);
			}
			setState(174);
			parameterDeclaration();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ParameterDeclarationContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public ParameterDeclarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_parameterDeclaration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterParameterDeclaration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitParameterDeclaration(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitParameterDeclaration(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ParameterDeclarationContext parameterDeclaration() throws RecognitionException {
		ParameterDeclarationContext _localctx = new ParameterDeclarationContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_parameterDeclaration);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(176);
			identifier();
			setState(177);
			match(T__8);
			setState(178);
			type();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IdListContext extends ParserRuleContext {
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public IdListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_idList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterIdList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitIdList(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitIdList(this);
			else return visitor.visitChildren(this);
		}
	}

	public final IdListContext idList() throws RecognitionException {
		IdListContext _localctx = new IdListContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_idList);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(185);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,13,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(180);
					identifier();
					setState(181);
					match(T__14);
					}
					} 
				}
				setState(187);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,13,_ctx);
			}
			setState(188);
			identifier();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class DatatypeDefinitionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public List<DatatypeBodyElementContext> datatypeBodyElement() {
			return getRuleContexts(DatatypeBodyElementContext.class);
		}
		public DatatypeBodyElementContext datatypeBodyElement(int i) {
			return getRuleContext(DatatypeBodyElementContext.class,i);
		}
		public DatatypeDefinitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_datatypeDefinition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterDatatypeDefinition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitDatatypeDefinition(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitDatatypeDefinition(this);
			else return visitor.visitChildren(this);
		}
	}

	public final DatatypeDefinitionContext datatypeDefinition() throws RecognitionException {
		DatatypeDefinitionContext _localctx = new DatatypeDefinitionContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_datatypeDefinition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(190);
			match(T__15);
			setState(191);
			identifier();
			setState(192);
			match(T__16);
			setState(193);
			match(T__3);
			setState(197);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while ((((_la) & ~0x3f) == 0 && ((1L << _la) & 268288L) != 0)) {
				{
				{
				setState(194);
				datatypeBodyElement();
				}
				}
				setState(199);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(200);
			match(T__4);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class DatatypeBodyElementContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public OperationDefinitionContext operationDefinition() {
			return getRuleContext(OperationDefinitionContext.class,0);
		}
		public DatatypeBodyElementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_datatypeBodyElement; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterDatatypeBodyElement(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitDatatypeBodyElement(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitDatatypeBodyElement(this);
			else return visitor.visitChildren(this);
		}
	}

	public final DatatypeBodyElementContext datatypeBodyElement() throws RecognitionException {
		DatatypeBodyElementContext _localctx = new DatatypeBodyElementContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_datatypeBodyElement);
		try {
			setState(209);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__17:
				enterOuterAlt(_localctx, 1);
				{
				setState(202);
				match(T__17);
				setState(203);
				identifier();
				setState(204);
				match(T__8);
				setState(205);
				type();
				setState(206);
				match(T__9);
				}
				break;
			case T__10:
			case T__11:
				enterOuterAlt(_localctx, 2);
				{
				setState(208);
				operationDefinition();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EnumerationContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public List<EnumerationLiteralContext> enumerationLiteral() {
			return getRuleContexts(EnumerationLiteralContext.class);
		}
		public EnumerationLiteralContext enumerationLiteral(int i) {
			return getRuleContext(EnumerationLiteralContext.class,i);
		}
		public EnumerationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_enumeration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterEnumeration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitEnumeration(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitEnumeration(this);
			else return visitor.visitChildren(this);
		}
	}

	public final EnumerationContext enumeration() throws RecognitionException {
		EnumerationContext _localctx = new EnumerationContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_enumeration);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(211);
			match(T__18);
			setState(212);
			identifier();
			setState(213);
			match(T__3);
			setState(215); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(214);
				enumerationLiteral();
				}
				}
				setState(217); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( _la==T__19 );
			setState(219);
			match(T__4);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EnumerationLiteralContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public EnumerationLiteralContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_enumerationLiteral; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterEnumerationLiteral(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitEnumerationLiteral(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitEnumerationLiteral(this);
			else return visitor.visitChildren(this);
		}
	}

	public final EnumerationLiteralContext enumerationLiteral() throws RecognitionException {
		EnumerationLiteralContext _localctx = new EnumerationLiteralContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_enumerationLiteral);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(221);
			match(T__19);
			setState(222);
			identifier();
			setState(223);
			match(T__9);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssociationContext extends ParserRuleContext {
		public AssociationEndContext associationEndA;
		public AssociationEndContext associationEndB;
		public List<AssociationEndContext> associationEnd() {
			return getRuleContexts(AssociationEndContext.class);
		}
		public AssociationEndContext associationEnd(int i) {
			return getRuleContext(AssociationEndContext.class,i);
		}
		public AssociationNameContext associationName() {
			return getRuleContext(AssociationNameContext.class,0);
		}
		public AssociationClassContext associationClass() {
			return getRuleContext(AssociationClassContext.class,0);
		}
		public AssociationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_association; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterAssociation(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitAssociation(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitAssociation(this);
			else return visitor.visitChildren(this);
		}
	}

	public final AssociationContext association() throws RecognitionException {
		AssociationContext _localctx = new AssociationContext(_ctx, getState());
		enterRule(_localctx, 30, RULE_association);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(225);
			match(T__20);
			setState(226);
			match(T__3);
			setState(228);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__22) {
				{
				setState(227);
				associationName();
				}
			}

			setState(230);
			((AssociationContext)_localctx).associationEndA = associationEnd();
			setState(231);
			((AssociationContext)_localctx).associationEndB = associationEnd();
			setState(233);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__21) {
				{
				setState(232);
				associationClass();
				}
			}

			setState(235);
			match(T__4);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssociationClassContext extends ParserRuleContext {
		public ClassBodyContext classBody() {
			return getRuleContext(ClassBodyContext.class,0);
		}
		public AssociationClassContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_associationClass; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterAssociationClass(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitAssociationClass(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitAssociationClass(this);
			else return visitor.visitChildren(this);
		}
	}

	public final AssociationClassContext associationClass() throws RecognitionException {
		AssociationClassContext _localctx = new AssociationClassContext(_ctx, getState());
		enterRule(_localctx, 32, RULE_associationClass);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(237);
			match(T__21);
			setState(238);
			match(T__3);
			setState(239);
			classBody();
			setState(240);
			match(T__4);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssociationNameContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public AssociationNameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_associationName; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterAssociationName(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitAssociationName(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitAssociationName(this);
			else return visitor.visitChildren(this);
		}
	}

	public final AssociationNameContext associationName() throws RecognitionException {
		AssociationNameContext _localctx = new AssociationNameContext(_ctx, getState());
		enterRule(_localctx, 34, RULE_associationName);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(242);
			match(T__22);
			setState(243);
			match(T__16);
			setState(244);
			identifier();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AssociationEndContext extends ParserRuleContext {
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public MultiplicityContext multiplicity() {
			return getRuleContext(MultiplicityContext.class,0);
		}
		public ConstraintsContext constraints() {
			return getRuleContext(ConstraintsContext.class,0);
		}
		public AssociationEndContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_associationEnd; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterAssociationEnd(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitAssociationEnd(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitAssociationEnd(this);
			else return visitor.visitChildren(this);
		}
	}

	public final AssociationEndContext associationEnd() throws RecognitionException {
		AssociationEndContext _localctx = new AssociationEndContext(_ctx, getState());
		enterRule(_localctx, 36, RULE_associationEnd);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(246);
			match(T__23);
			setState(247);
			match(T__16);
			setState(248);
			identifier();
			setState(250);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 3758096384L) != 0) || _la==INT) {
				{
				setState(249);
				multiplicity();
				}
			}

			setState(253);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__24 || _la==T__25) {
				{
				setState(252);
				_la = _input.LA(1);
				if ( !(_la==T__24 || _la==T__25) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				}
			}

			setState(257);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__26 || _la==T__27) {
				{
				setState(255);
				_la = _input.LA(1);
				if ( !(_la==T__26 || _la==T__27) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(256);
				identifier();
				}
			}

			setState(260);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__3) {
				{
				setState(259);
				constraints();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class MultiplicityContext extends ParserRuleContext {
		public List<TerminalNode> INT() { return getTokens(SimpleClassModelWithConstraintsParser.INT); }
		public TerminalNode INT(int i) {
			return getToken(SimpleClassModelWithConstraintsParser.INT, i);
		}
		public MultiplicityContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_multiplicity; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterMultiplicity(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitMultiplicity(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitMultiplicity(this);
			else return visitor.visitChildren(this);
		}
	}

	public final MultiplicityContext multiplicity() throws RecognitionException {
		MultiplicityContext _localctx = new MultiplicityContext(_ctx, getState());
		enterRule(_localctx, 38, RULE_multiplicity);
		try {
			setState(272);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,23,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(262);
				match(T__28);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(263);
				match(T__29);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(264);
				match(T__30);
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(265);
				match(INT);
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(266);
				match(INT);
				setState(267);
				match(T__31);
				setState(268);
				match(INT);
				}
				break;
			case 6:
				enterOuterAlt(_localctx, 6);
				{
				setState(269);
				match(INT);
				setState(270);
				match(T__31);
				setState(271);
				match(T__28);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ConstraintsContext extends ParserRuleContext {
		public ConstraintsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_constraints; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterConstraints(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitConstraints(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitConstraints(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ConstraintsContext constraints() throws RecognitionException {
		ConstraintsContext _localctx = new ConstraintsContext(_ctx, getState());
		enterRule(_localctx, 40, RULE_constraints);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(274);
			match(T__3);
			setState(276); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(275);
				_la = _input.LA(1);
				if ( !(_la==T__32 || _la==T__33) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				}
				}
				setState(278); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( _la==T__32 || _la==T__33 );
			setState(280);
			match(T__4);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IdentifierContext extends ParserRuleContext {
		public TerminalNode ID() { return getToken(SimpleClassModelWithConstraintsParser.ID, 0); }
		public IdentifierContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_identifier; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterIdentifier(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitIdentifier(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitIdentifier(this);
			else return visitor.visitChildren(this);
		}
	}

	public final IdentifierContext identifier() throws RecognitionException {
		IdentifierContext _localctx = new IdentifierContext(_ctx, getState());
		enterRule(_localctx, 42, RULE_identifier);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(282);
			match(ID);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TypeContext extends ParserRuleContext {
		public PrimitiveTypeContext primitiveType() {
			return getRuleContext(PrimitiveTypeContext.class,0);
		}
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public TypeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitType(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitType(this);
			else return visitor.visitChildren(this);
		}
	}

	public final TypeContext type() throws RecognitionException {
		TypeContext _localctx = new TypeContext(_ctx, getState());
		enterRule(_localctx, 44, RULE_type);
		try {
			setState(286);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__34:
			case T__35:
			case T__36:
			case T__37:
				enterOuterAlt(_localctx, 1);
				{
				setState(284);
				primitiveType();
				}
				break;
			case ID:
				enterOuterAlt(_localctx, 2);
				{
				setState(285);
				identifier();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PrimitiveTypeContext extends ParserRuleContext {
		public PrimitiveTypeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_primitiveType; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterPrimitiveType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitPrimitiveType(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitPrimitiveType(this);
			else return visitor.visitChildren(this);
		}
	}

	public final PrimitiveTypeContext primitiveType() throws RecognitionException {
		PrimitiveTypeContext _localctx = new PrimitiveTypeContext(_ctx, getState());
		enterRule(_localctx, 46, RULE_primitiveType);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(288);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 515396075520L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class MultipleContextSpecificationsContext extends ParserRuleContext {
		public List<InvariantContextContext> invariantContext() {
			return getRuleContexts(InvariantContextContext.class);
		}
		public InvariantContextContext invariantContext(int i) {
			return getRuleContext(InvariantContextContext.class,i);
		}
		public List<PrepostContextContext> prepostContext() {
			return getRuleContexts(PrepostContextContext.class);
		}
		public PrepostContextContext prepostContext(int i) {
			return getRuleContext(PrepostContextContext.class,i);
		}
		public MultipleContextSpecificationsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_multipleContextSpecifications; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterMultipleContextSpecifications(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitMultipleContextSpecifications(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitMultipleContextSpecifications(this);
			else return visitor.visitChildren(this);
		}
	}

	public final MultipleContextSpecificationsContext multipleContextSpecifications() throws RecognitionException {
		MultipleContextSpecificationsContext _localctx = new MultipleContextSpecificationsContext(_ctx, getState());
		enterRule(_localctx, 48, RULE_multipleContextSpecifications);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(292); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				setState(292);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,26,_ctx) ) {
				case 1:
					{
					setState(290);
					invariantContext();
					}
					break;
				case 2:
					{
					setState(291);
					prepostContext();
					}
					break;
				}
				}
				setState(294); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( _la==T__38 );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class InvariantContextContext extends ParserRuleContext {
		public TerminalNode ID() { return getToken(SimpleClassModelWithConstraintsParser.ID, 0); }
		public List<InvariantContext> invariant() {
			return getRuleContexts(InvariantContext.class);
		}
		public InvariantContext invariant(int i) {
			return getRuleContext(InvariantContext.class,i);
		}
		public InvariantContextContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_invariantContext; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterInvariantContext(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitInvariantContext(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitInvariantContext(this);
			else return visitor.visitChildren(this);
		}
	}

	public final InvariantContextContext invariantContext() throws RecognitionException {
		InvariantContextContext _localctx = new InvariantContextContext(_ctx, getState());
		enterRule(_localctx, 50, RULE_invariantContext);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(296);
			match(T__38);
			setState(297);
			match(ID);
			setState(299); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(298);
				invariant();
				}
				}
				setState(301); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( _la==T__39 );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class InvariantContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public TerminalNode ID() { return getToken(SimpleClassModelWithConstraintsParser.ID, 0); }
		public InvariantContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_invariant; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterInvariant(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitInvariant(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitInvariant(this);
			else return visitor.visitChildren(this);
		}
	}

	public final InvariantContext invariant() throws RecognitionException {
		InvariantContext _localctx = new InvariantContext(_ctx, getState());
		enterRule(_localctx, 52, RULE_invariant);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(303);
			match(T__39);
			setState(305);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==ID) {
				{
				setState(304);
				match(ID);
				}
			}

			setState(307);
			match(T__8);
			setState(308);
			expression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PrepostContextContext extends ParserRuleContext {
		public TerminalNode ID() { return getToken(SimpleClassModelWithConstraintsParser.ID, 0); }
		public OperationDeclarationContext operationDeclaration() {
			return getRuleContext(OperationDeclarationContext.class,0);
		}
		public PrepostSpecificationContext prepostSpecification() {
			return getRuleContext(PrepostSpecificationContext.class,0);
		}
		public PrepostContextContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_prepostContext; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterPrepostContext(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitPrepostContext(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitPrepostContext(this);
			else return visitor.visitChildren(this);
		}
	}

	public final PrepostContextContext prepostContext() throws RecognitionException {
		PrepostContextContext _localctx = new PrepostContextContext(_ctx, getState());
		enterRule(_localctx, 54, RULE_prepostContext);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(310);
			match(T__38);
			setState(311);
			match(ID);
			setState(312);
			match(T__40);
			setState(313);
			operationDeclaration();
			setState(314);
			prepostSpecification();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class OperationDeclarationContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public OcltypeContext ocltype() {
			return getRuleContext(OcltypeContext.class,0);
		}
		public ParameterDeclarationsContext parameterDeclarations() {
			return getRuleContext(ParameterDeclarationsContext.class,0);
		}
		public OperationDeclarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_operationDeclaration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterOperationDeclaration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitOperationDeclaration(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitOperationDeclaration(this);
			else return visitor.visitChildren(this);
		}
	}

	public final OperationDeclarationContext operationDeclaration() throws RecognitionException {
		OperationDeclarationContext _localctx = new OperationDeclarationContext(_ctx, getState());
		enterRule(_localctx, 56, RULE_operationDeclaration);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(316);
			identifier();
			setState(317);
			match(T__12);
			setState(319);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==ID) {
				{
				setState(318);
				parameterDeclarations();
				}
			}

			setState(321);
			match(T__13);
			setState(322);
			match(T__8);
			setState(323);
			ocltype();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PrepostSpecificationContext extends ParserRuleContext {
		public List<PreconditionContext> precondition() {
			return getRuleContexts(PreconditionContext.class);
		}
		public PreconditionContext precondition(int i) {
			return getRuleContext(PreconditionContext.class,i);
		}
		public List<PostconditionContext> postcondition() {
			return getRuleContexts(PostconditionContext.class);
		}
		public PostconditionContext postcondition(int i) {
			return getRuleContext(PostconditionContext.class,i);
		}
		public PrepostSpecificationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_prepostSpecification; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterPrepostSpecification(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitPrepostSpecification(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitPrepostSpecification(this);
			else return visitor.visitChildren(this);
		}
	}

	public final PrepostSpecificationContext prepostSpecification() throws RecognitionException {
		PrepostSpecificationContext _localctx = new PrepostSpecificationContext(_ctx, getState());
		enterRule(_localctx, 58, RULE_prepostSpecification);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(328);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__41) {
				{
				{
				setState(325);
				precondition();
				}
				}
				setState(330);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(334);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__42) {
				{
				{
				setState(331);
				postcondition();
				}
				}
				setState(336);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PreconditionContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public PreconditionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_precondition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterPrecondition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitPrecondition(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitPrecondition(this);
			else return visitor.visitChildren(this);
		}
	}

	public final PreconditionContext precondition() throws RecognitionException {
		PreconditionContext _localctx = new PreconditionContext(_ctx, getState());
		enterRule(_localctx, 60, RULE_precondition);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(337);
			match(T__41);
			setState(338);
			expression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PostconditionContext extends ParserRuleContext {
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public PostconditionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_postcondition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterPostcondition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitPostcondition(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitPostcondition(this);
			else return visitor.visitChildren(this);
		}
	}

	public final PostconditionContext postcondition() throws RecognitionException {
		PostconditionContext _localctx = new PostconditionContext(_ctx, getState());
		enterRule(_localctx, 62, RULE_postcondition);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(340);
			match(T__42);
			setState(341);
			expression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class OcltypeContext extends ParserRuleContext {
		public List<OcltypeContext> ocltype() {
			return getRuleContexts(OcltypeContext.class);
		}
		public OcltypeContext ocltype(int i) {
			return getRuleContext(OcltypeContext.class,i);
		}
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public OcltypeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_ocltype; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterOcltype(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitOcltype(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitOcltype(this);
			else return visitor.visitChildren(this);
		}
	}

	public final OcltypeContext ocltype() throws RecognitionException {
		OcltypeContext _localctx = new OcltypeContext(_ctx, getState());
		enterRule(_localctx, 64, RULE_ocltype);
		try {
			setState(383);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__43:
				enterOuterAlt(_localctx, 1);
				{
				setState(343);
				match(T__43);
				setState(344);
				match(T__12);
				setState(345);
				ocltype();
				setState(346);
				match(T__13);
				}
				break;
			case T__44:
				enterOuterAlt(_localctx, 2);
				{
				setState(348);
				match(T__44);
				setState(349);
				match(T__12);
				setState(350);
				ocltype();
				setState(351);
				match(T__13);
				}
				break;
			case T__45:
				enterOuterAlt(_localctx, 3);
				{
				setState(353);
				match(T__45);
				setState(354);
				match(T__12);
				setState(355);
				ocltype();
				setState(356);
				match(T__13);
				}
				break;
			case T__46:
				enterOuterAlt(_localctx, 4);
				{
				setState(358);
				match(T__46);
				setState(359);
				match(T__12);
				setState(360);
				ocltype();
				setState(361);
				match(T__13);
				}
				break;
			case T__47:
				enterOuterAlt(_localctx, 5);
				{
				setState(363);
				match(T__47);
				setState(364);
				match(T__12);
				setState(365);
				ocltype();
				setState(366);
				match(T__13);
				}
				break;
			case T__48:
				enterOuterAlt(_localctx, 6);
				{
				setState(368);
				match(T__48);
				setState(369);
				match(T__12);
				setState(370);
				ocltype();
				setState(371);
				match(T__14);
				setState(372);
				ocltype();
				setState(373);
				match(T__13);
				}
				break;
			case T__49:
				enterOuterAlt(_localctx, 7);
				{
				setState(375);
				match(T__49);
				setState(376);
				match(T__12);
				setState(377);
				ocltype();
				setState(378);
				match(T__14);
				setState(379);
				ocltype();
				setState(380);
				match(T__13);
				}
				break;
			case T__34:
			case T__35:
			case T__36:
			case T__37:
			case ID:
				enterOuterAlt(_localctx, 8);
				{
				setState(382);
				type();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionListContext extends ParserRuleContext {
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public ExpressionListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expressionList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterExpressionList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitExpressionList(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitExpressionList(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ExpressionListContext expressionList() throws RecognitionException {
		ExpressionListContext _localctx = new ExpressionListContext(_ctx, getState());
		enterRule(_localctx, 66, RULE_expressionList);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(390);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,34,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(385);
					expression();
					setState(386);
					match(T__14);
					}
					} 
				}
				setState(392);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,34,_ctx);
			}
			setState(393);
			expression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionContext extends ParserRuleContext {
		public LogicalExpressionContext logicalExpression() {
			return getRuleContext(LogicalExpressionContext.class,0);
		}
		public ConditionalExpressionContext conditionalExpression() {
			return getRuleContext(ConditionalExpressionContext.class,0);
		}
		public LambdaExpressionContext lambdaExpression() {
			return getRuleContext(LambdaExpressionContext.class,0);
		}
		public LetExpressionContext letExpression() {
			return getRuleContext(LetExpressionContext.class,0);
		}
		public ExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ExpressionContext expression() throws RecognitionException {
		ExpressionContext _localctx = new ExpressionContext(_ctx, getState());
		enterRule(_localctx, 68, RULE_expression);
		try {
			setState(399);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__12:
			case T__26:
			case T__27:
			case T__61:
			case T__79:
			case T__80:
			case T__184:
			case T__185:
			case T__186:
			case T__187:
			case T__188:
			case INT:
			case FLOAT_LITERAL:
			case STRING1_LITERAL:
			case STRING2_LITERAL:
			case ENUMERATION_LITERAL:
			case NULL_LITERAL:
			case ID:
				enterOuterAlt(_localctx, 1);
				{
				setState(395);
				logicalExpression(0);
				}
				break;
			case T__54:
				enterOuterAlt(_localctx, 2);
				{
				setState(396);
				conditionalExpression();
				}
				break;
			case T__58:
				enterOuterAlt(_localctx, 3);
				{
				setState(397);
				lambdaExpression();
				}
				break;
			case T__60:
				enterOuterAlt(_localctx, 4);
				{
				setState(398);
				letExpression();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class BasicExpressionContext extends ParserRuleContext {
		public TerminalNode NULL_LITERAL() { return getToken(SimpleClassModelWithConstraintsParser.NULL_LITERAL, 0); }
		public TerminalNode ID() { return getToken(SimpleClassModelWithConstraintsParser.ID, 0); }
		public TerminalNode INT() { return getToken(SimpleClassModelWithConstraintsParser.INT, 0); }
		public TerminalNode FLOAT_LITERAL() { return getToken(SimpleClassModelWithConstraintsParser.FLOAT_LITERAL, 0); }
		public TerminalNode STRING1_LITERAL() { return getToken(SimpleClassModelWithConstraintsParser.STRING1_LITERAL, 0); }
		public TerminalNode STRING2_LITERAL() { return getToken(SimpleClassModelWithConstraintsParser.STRING2_LITERAL, 0); }
		public TerminalNode ENUMERATION_LITERAL() { return getToken(SimpleClassModelWithConstraintsParser.ENUMERATION_LITERAL, 0); }
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public BasicExpressionContext basicExpression() {
			return getRuleContext(BasicExpressionContext.class,0);
		}
		public ExpressionListContext expressionList() {
			return getRuleContext(ExpressionListContext.class,0);
		}
		public BasicExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_basicExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterBasicExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitBasicExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitBasicExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final BasicExpressionContext basicExpression() throws RecognitionException {
		return basicExpression(0);
	}

	private BasicExpressionContext basicExpression(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		BasicExpressionContext _localctx = new BasicExpressionContext(_ctx, _parentState);
		BasicExpressionContext _prevctx = _localctx;
		int _startState = 70;
		enterRecursionRule(_localctx, 70, RULE_basicExpression, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(415);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,36,_ctx) ) {
			case 1:
				{
				setState(402);
				match(NULL_LITERAL);
				}
				break;
			case 2:
				{
				setState(403);
				match(ID);
				setState(404);
				match(T__53);
				}
				break;
			case 3:
				{
				setState(405);
				match(INT);
				}
				break;
			case 4:
				{
				setState(406);
				match(FLOAT_LITERAL);
				}
				break;
			case 5:
				{
				setState(407);
				match(STRING1_LITERAL);
				}
				break;
			case 6:
				{
				setState(408);
				match(STRING2_LITERAL);
				}
				break;
			case 7:
				{
				setState(409);
				match(ENUMERATION_LITERAL);
				}
				break;
			case 8:
				{
				setState(410);
				match(ID);
				}
				break;
			case 9:
				{
				setState(411);
				match(T__12);
				setState(412);
				expression();
				setState(413);
				match(T__13);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(440);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,41,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(438);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,40,_ctx) ) {
					case 1:
						{
						_localctx = new BasicExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_basicExpression);
						setState(417);
						if (!(precpred(_ctx, 11))) throw new FailedPredicateException(this, "precpred(_ctx, 11)");
						setState(418);
						match(T__50);
						setState(419);
						match(ID);
						setState(425);
						_errHandler.sync(this);
						switch ( getInterpreter().adaptivePredict(_input,38,_ctx) ) {
						case 1:
							{
							setState(420);
							match(T__12);
							setState(422);
							_errHandler.sync(this);
							_la = _input.LA(1);
							if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7530018577366130688L) != 0) || _la==T__79 || _la==T__80 || ((((_la - 185)) & ~0x3f) == 0 && ((1L << (_la - 185)) & 4095L) != 0)) {
								{
								setState(421);
								expressionList();
								}
							}

							setState(424);
							match(T__13);
							}
							break;
						}
						}
						break;
					case 2:
						{
						_localctx = new BasicExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_basicExpression);
						setState(427);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(428);
						match(T__12);
						setState(430);
						_errHandler.sync(this);
						_la = _input.LA(1);
						if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7530018577366130688L) != 0) || _la==T__79 || _la==T__80 || ((((_la - 185)) & ~0x3f) == 0 && ((1L << (_la - 185)) & 4095L) != 0)) {
							{
							setState(429);
							expressionList();
							}
						}

						setState(432);
						match(T__13);
						}
						break;
					case 3:
						{
						_localctx = new BasicExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_basicExpression);
						setState(433);
						if (!(precpred(_ctx, 9))) throw new FailedPredicateException(this, "precpred(_ctx, 9)");
						setState(434);
						match(T__51);
						setState(435);
						expression();
						setState(436);
						match(T__52);
						}
						break;
					}
					} 
				}
				setState(442);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,41,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ConditionalExpressionContext extends ParserRuleContext {
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public ConditionalExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_conditionalExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterConditionalExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitConditionalExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitConditionalExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final ConditionalExpressionContext conditionalExpression() throws RecognitionException {
		ConditionalExpressionContext _localctx = new ConditionalExpressionContext(_ctx, getState());
		enterRule(_localctx, 72, RULE_conditionalExpression);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(443);
			match(T__54);
			setState(444);
			expression();
			setState(445);
			match(T__55);
			setState(446);
			expression();
			setState(447);
			match(T__56);
			setState(448);
			expression();
			setState(449);
			match(T__57);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class LambdaExpressionContext extends ParserRuleContext {
		public IdentifierContext identifier() {
			return getRuleContext(IdentifierContext.class,0);
		}
		public OcltypeContext ocltype() {
			return getRuleContext(OcltypeContext.class,0);
		}
		public ExpressionContext expression() {
			return getRuleContext(ExpressionContext.class,0);
		}
		public LambdaExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_lambdaExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterLambdaExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitLambdaExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitLambdaExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final LambdaExpressionContext lambdaExpression() throws RecognitionException {
		LambdaExpressionContext _localctx = new LambdaExpressionContext(_ctx, getState());
		enterRule(_localctx, 74, RULE_lambdaExpression);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(451);
			match(T__58);
			setState(452);
			identifier();
			setState(453);
			match(T__8);
			setState(454);
			ocltype();
			setState(455);
			match(T__59);
			setState(456);
			expression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class LetExpressionContext extends ParserRuleContext {
		public TerminalNode ID() { return getToken(SimpleClassModelWithConstraintsParser.ID, 0); }
		public OcltypeContext ocltype() {
			return getRuleContext(OcltypeContext.class,0);
		}
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public LetExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_letExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterLetExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitLetExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitLetExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final LetExpressionContext letExpression() throws RecognitionException {
		LetExpressionContext _localctx = new LetExpressionContext(_ctx, getState());
		enterRule(_localctx, 76, RULE_letExpression);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(458);
			match(T__60);
			setState(459);
			match(ID);
			setState(460);
			match(T__8);
			setState(461);
			ocltype();
			setState(462);
			match(T__16);
			setState(463);
			expression();
			setState(464);
			match(T__59);
			setState(465);
			expression();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class LogicalExpressionContext extends ParserRuleContext {
		public List<LogicalExpressionContext> logicalExpression() {
			return getRuleContexts(LogicalExpressionContext.class);
		}
		public LogicalExpressionContext logicalExpression(int i) {
			return getRuleContext(LogicalExpressionContext.class,i);
		}
		public EqualityExpressionContext equalityExpression() {
			return getRuleContext(EqualityExpressionContext.class,0);
		}
		public LogicalExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_logicalExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterLogicalExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitLogicalExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitLogicalExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final LogicalExpressionContext logicalExpression() throws RecognitionException {
		return logicalExpression(0);
	}

	private LogicalExpressionContext logicalExpression(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		LogicalExpressionContext _localctx = new LogicalExpressionContext(_ctx, _parentState);
		LogicalExpressionContext _prevctx = _localctx;
		int _startState = 78;
		enterRecursionRule(_localctx, 78, RULE_logicalExpression, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(471);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__61:
				{
				setState(468);
				match(T__61);
				setState(469);
				logicalExpression(8);
				}
				break;
			case T__12:
			case T__26:
			case T__27:
			case T__79:
			case T__80:
			case T__184:
			case T__185:
			case T__186:
			case T__187:
			case T__188:
			case INT:
			case FLOAT_LITERAL:
			case STRING1_LITERAL:
			case STRING2_LITERAL:
			case ENUMERATION_LITERAL:
			case NULL_LITERAL:
			case ID:
				{
				setState(470);
				equalityExpression();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(493);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,44,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(491);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,43,_ctx) ) {
					case 1:
						{
						_localctx = new LogicalExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_logicalExpression);
						setState(473);
						if (!(precpred(_ctx, 7))) throw new FailedPredicateException(this, "precpred(_ctx, 7)");
						setState(474);
						match(T__62);
						setState(475);
						logicalExpression(8);
						}
						break;
					case 2:
						{
						_localctx = new LogicalExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_logicalExpression);
						setState(476);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(477);
						match(T__63);
						setState(478);
						logicalExpression(7);
						}
						break;
					case 3:
						{
						_localctx = new LogicalExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_logicalExpression);
						setState(479);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(480);
						match(T__64);
						setState(481);
						logicalExpression(6);
						}
						break;
					case 4:
						{
						_localctx = new LogicalExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_logicalExpression);
						setState(482);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(483);
						match(T__65);
						setState(484);
						logicalExpression(5);
						}
						break;
					case 5:
						{
						_localctx = new LogicalExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_logicalExpression);
						setState(485);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(486);
						match(T__66);
						setState(487);
						logicalExpression(4);
						}
						break;
					case 6:
						{
						_localctx = new LogicalExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_logicalExpression);
						setState(488);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(489);
						match(T__67);
						setState(490);
						logicalExpression(3);
						}
						break;
					}
					} 
				}
				setState(495);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,44,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class EqualityExpressionContext extends ParserRuleContext {
		public List<AdditiveExpressionContext> additiveExpression() {
			return getRuleContexts(AdditiveExpressionContext.class);
		}
		public AdditiveExpressionContext additiveExpression(int i) {
			return getRuleContext(AdditiveExpressionContext.class,i);
		}
		public EqualityExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_equalityExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterEqualityExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitEqualityExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitEqualityExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final EqualityExpressionContext equalityExpression() throws RecognitionException {
		EqualityExpressionContext _localctx = new EqualityExpressionContext(_ctx, getState());
		enterRule(_localctx, 80, RULE_equalityExpression);
		int _la;
		try {
			setState(501);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,45,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(496);
				additiveExpression(0);
				setState(497);
				_la = _input.LA(1);
				if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 33686016L) != 0) || ((((_la - 69)) & ~0x3f) == 0 && ((1L << (_la - 69)) & 127L) != 0)) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(498);
				additiveExpression(0);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(500);
				additiveExpression(0);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AdditiveExpressionContext extends ParserRuleContext {
		public FactorExpressionContext factorExpression() {
			return getRuleContext(FactorExpressionContext.class,0);
		}
		public List<AdditiveExpressionContext> additiveExpression() {
			return getRuleContexts(AdditiveExpressionContext.class);
		}
		public AdditiveExpressionContext additiveExpression(int i) {
			return getRuleContext(AdditiveExpressionContext.class,i);
		}
		public AdditiveExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_additiveExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterAdditiveExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitAdditiveExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitAdditiveExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final AdditiveExpressionContext additiveExpression() throws RecognitionException {
		return additiveExpression(0);
	}

	private AdditiveExpressionContext additiveExpression(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		AdditiveExpressionContext _localctx = new AdditiveExpressionContext(_ctx, _parentState);
		AdditiveExpressionContext _prevctx = _localctx;
		int _startState = 82;
		enterRecursionRule(_localctx, 82, RULE_additiveExpression, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(504);
			factorExpression();
			}
			_ctx.stop = _input.LT(-1);
			setState(514);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,47,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(512);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,46,_ctx) ) {
					case 1:
						{
						_localctx = new AdditiveExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_additiveExpression);
						setState(506);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(507);
						_la = _input.LA(1);
						if ( !(_la==T__26 || _la==T__27) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(508);
						additiveExpression(4);
						}
						break;
					case 2:
						{
						_localctx = new AdditiveExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_additiveExpression);
						setState(509);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(510);
						_la = _input.LA(1);
						if ( !(((((_la - 32)) & ~0x3f) == 0 && ((1L << (_la - 32)) & 17592186568705L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(511);
						additiveExpression(3);
						}
						break;
					}
					} 
				}
				setState(516);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,47,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FactorExpressionContext extends ParserRuleContext {
		public Factor2ExpressionContext factor2Expression() {
			return getRuleContext(Factor2ExpressionContext.class,0);
		}
		public FactorExpressionContext factorExpression() {
			return getRuleContext(FactorExpressionContext.class,0);
		}
		public FactorExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_factorExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterFactorExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitFactorExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitFactorExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final FactorExpressionContext factorExpression() throws RecognitionException {
		FactorExpressionContext _localctx = new FactorExpressionContext(_ctx, getState());
		enterRule(_localctx, 84, RULE_factorExpression);
		int _la;
		try {
			setState(522);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,48,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(517);
				factor2Expression(0);
				setState(518);
				_la = _input.LA(1);
				if ( !(((((_la - 29)) & ~0x3f) == 0 && ((1L << (_la - 29)) & 1970324836974593L) != 0)) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(519);
				factorExpression();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(521);
				factor2Expression(0);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Factor2ExpressionContext extends ParserRuleContext {
		public Factor2ExpressionContext factor2Expression() {
			return getRuleContext(Factor2ExpressionContext.class,0);
		}
		public SetExpressionContext setExpression() {
			return getRuleContext(SetExpressionContext.class,0);
		}
		public BasicExpressionContext basicExpression() {
			return getRuleContext(BasicExpressionContext.class,0);
		}
		public List<ExpressionContext> expression() {
			return getRuleContexts(ExpressionContext.class);
		}
		public ExpressionContext expression(int i) {
			return getRuleContext(ExpressionContext.class,i);
		}
		public IdentOptTypeContext identOptType() {
			return getRuleContext(IdentOptTypeContext.class,0);
		}
		public List<IdentifierContext> identifier() {
			return getRuleContexts(IdentifierContext.class);
		}
		public IdentifierContext identifier(int i) {
			return getRuleContext(IdentifierContext.class,i);
		}
		public Factor2ExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_factor2Expression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterFactor2Expression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitFactor2Expression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitFactor2Expression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final Factor2ExpressionContext factor2Expression() throws RecognitionException {
		return factor2Expression(0);
	}

	private Factor2ExpressionContext factor2Expression(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		Factor2ExpressionContext _localctx = new Factor2ExpressionContext(_ctx, _parentState);
		Factor2ExpressionContext _prevctx = _localctx;
		int _startState = 86;
		enterRecursionRule(_localctx, 86, RULE_factor2Expression, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(529);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__26:
			case T__27:
			case T__79:
			case T__80:
				{
				setState(525);
				_la = _input.LA(1);
				if ( !(((((_la - 27)) & ~0x3f) == 0 && ((1L << (_la - 27)) & 27021597764222979L) != 0)) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(526);
				factor2Expression(73);
				}
				break;
			case T__184:
			case T__185:
			case T__186:
			case T__187:
			case T__188:
				{
				setState(527);
				setExpression();
				}
				break;
			case T__12:
			case INT:
			case FLOAT_LITERAL:
			case STRING1_LITERAL:
			case STRING2_LITERAL:
			case ENUMERATION_LITERAL:
			case NULL_LITERAL:
			case ID:
				{
				setState(528);
				basicExpression(0);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(817);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,51,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(815);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,50,_ctx) ) {
					case 1:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(531);
						if (!(precpred(_ctx, 72))) throw new FailedPredicateException(this, "precpred(_ctx, 72)");
						setState(532);
						match(T__81);
						}
						break;
					case 2:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(533);
						if (!(precpred(_ctx, 71))) throw new FailedPredicateException(this, "precpred(_ctx, 71)");
						setState(534);
						match(T__82);
						}
						break;
					case 3:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(535);
						if (!(precpred(_ctx, 70))) throw new FailedPredicateException(this, "precpred(_ctx, 70)");
						setState(536);
						_la = _input.LA(1);
						if ( !(((((_la - 84)) & ~0x3f) == 0 && ((1L << (_la - 84)) & 127L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						}
						break;
					case 4:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(537);
						if (!(precpred(_ctx, 69))) throw new FailedPredicateException(this, "precpred(_ctx, 69)");
						setState(538);
						match(T__90);
						}
						break;
					case 5:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(539);
						if (!(precpred(_ctx, 68))) throw new FailedPredicateException(this, "precpred(_ctx, 68)");
						setState(540);
						match(T__91);
						}
						break;
					case 6:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(541);
						if (!(precpred(_ctx, 67))) throw new FailedPredicateException(this, "precpred(_ctx, 67)");
						setState(542);
						match(T__92);
						}
						break;
					case 7:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(543);
						if (!(precpred(_ctx, 66))) throw new FailedPredicateException(this, "precpred(_ctx, 66)");
						setState(544);
						match(T__93);
						}
						break;
					case 8:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(545);
						if (!(precpred(_ctx, 65))) throw new FailedPredicateException(this, "precpred(_ctx, 65)");
						setState(546);
						match(T__94);
						}
						break;
					case 9:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(547);
						if (!(precpred(_ctx, 64))) throw new FailedPredicateException(this, "precpred(_ctx, 64)");
						setState(548);
						match(T__95);
						}
						break;
					case 10:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(549);
						if (!(precpred(_ctx, 63))) throw new FailedPredicateException(this, "precpred(_ctx, 63)");
						setState(550);
						match(T__96);
						}
						break;
					case 11:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(551);
						if (!(precpred(_ctx, 62))) throw new FailedPredicateException(this, "precpred(_ctx, 62)");
						setState(552);
						match(T__97);
						}
						break;
					case 12:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(553);
						if (!(precpred(_ctx, 61))) throw new FailedPredicateException(this, "precpred(_ctx, 61)");
						setState(554);
						match(T__98);
						}
						break;
					case 13:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(555);
						if (!(precpred(_ctx, 60))) throw new FailedPredicateException(this, "precpred(_ctx, 60)");
						setState(556);
						match(T__99);
						}
						break;
					case 14:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(557);
						if (!(precpred(_ctx, 59))) throw new FailedPredicateException(this, "precpred(_ctx, 59)");
						setState(558);
						match(T__100);
						}
						break;
					case 15:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(559);
						if (!(precpred(_ctx, 58))) throw new FailedPredicateException(this, "precpred(_ctx, 58)");
						setState(560);
						match(T__101);
						}
						break;
					case 16:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(561);
						if (!(precpred(_ctx, 57))) throw new FailedPredicateException(this, "precpred(_ctx, 57)");
						setState(562);
						match(T__102);
						}
						break;
					case 17:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(563);
						if (!(precpred(_ctx, 56))) throw new FailedPredicateException(this, "precpred(_ctx, 56)");
						setState(564);
						match(T__103);
						}
						break;
					case 18:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(565);
						if (!(precpred(_ctx, 55))) throw new FailedPredicateException(this, "precpred(_ctx, 55)");
						setState(566);
						match(T__104);
						}
						break;
					case 19:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(567);
						if (!(precpred(_ctx, 54))) throw new FailedPredicateException(this, "precpred(_ctx, 54)");
						setState(568);
						match(T__105);
						}
						break;
					case 20:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(569);
						if (!(precpred(_ctx, 53))) throw new FailedPredicateException(this, "precpred(_ctx, 53)");
						setState(570);
						match(T__106);
						}
						break;
					case 21:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(571);
						if (!(precpred(_ctx, 52))) throw new FailedPredicateException(this, "precpred(_ctx, 52)");
						setState(572);
						match(T__107);
						}
						break;
					case 22:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(573);
						if (!(precpred(_ctx, 51))) throw new FailedPredicateException(this, "precpred(_ctx, 51)");
						setState(574);
						match(T__108);
						}
						break;
					case 23:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(575);
						if (!(precpred(_ctx, 50))) throw new FailedPredicateException(this, "precpred(_ctx, 50)");
						setState(576);
						match(T__109);
						}
						break;
					case 24:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(577);
						if (!(precpred(_ctx, 49))) throw new FailedPredicateException(this, "precpred(_ctx, 49)");
						setState(578);
						match(T__110);
						}
						break;
					case 25:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(579);
						if (!(precpred(_ctx, 48))) throw new FailedPredicateException(this, "precpred(_ctx, 48)");
						setState(580);
						match(T__111);
						}
						break;
					case 26:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(581);
						if (!(precpred(_ctx, 47))) throw new FailedPredicateException(this, "precpred(_ctx, 47)");
						setState(582);
						match(T__112);
						}
						break;
					case 27:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(583);
						if (!(precpred(_ctx, 46))) throw new FailedPredicateException(this, "precpred(_ctx, 46)");
						setState(584);
						match(T__113);
						}
						break;
					case 28:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(585);
						if (!(precpred(_ctx, 45))) throw new FailedPredicateException(this, "precpred(_ctx, 45)");
						setState(586);
						match(T__114);
						}
						break;
					case 29:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(587);
						if (!(precpred(_ctx, 44))) throw new FailedPredicateException(this, "precpred(_ctx, 44)");
						setState(588);
						match(T__115);
						}
						break;
					case 30:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(589);
						if (!(precpred(_ctx, 43))) throw new FailedPredicateException(this, "precpred(_ctx, 43)");
						setState(590);
						match(T__116);
						}
						break;
					case 31:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(591);
						if (!(precpred(_ctx, 42))) throw new FailedPredicateException(this, "precpred(_ctx, 42)");
						setState(592);
						match(T__117);
						}
						break;
					case 32:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(593);
						if (!(precpred(_ctx, 41))) throw new FailedPredicateException(this, "precpred(_ctx, 41)");
						setState(594);
						match(T__118);
						}
						break;
					case 33:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(595);
						if (!(precpred(_ctx, 40))) throw new FailedPredicateException(this, "precpred(_ctx, 40)");
						setState(596);
						match(T__119);
						}
						break;
					case 34:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(597);
						if (!(precpred(_ctx, 39))) throw new FailedPredicateException(this, "precpred(_ctx, 39)");
						setState(598);
						match(T__120);
						}
						break;
					case 35:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(599);
						if (!(precpred(_ctx, 38))) throw new FailedPredicateException(this, "precpred(_ctx, 38)");
						setState(600);
						match(T__121);
						}
						break;
					case 36:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(601);
						if (!(precpred(_ctx, 37))) throw new FailedPredicateException(this, "precpred(_ctx, 37)");
						setState(602);
						match(T__122);
						}
						break;
					case 37:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(603);
						if (!(precpred(_ctx, 36))) throw new FailedPredicateException(this, "precpred(_ctx, 36)");
						setState(604);
						match(T__123);
						}
						break;
					case 38:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(605);
						if (!(precpred(_ctx, 35))) throw new FailedPredicateException(this, "precpred(_ctx, 35)");
						setState(606);
						match(T__124);
						}
						break;
					case 39:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(607);
						if (!(precpred(_ctx, 34))) throw new FailedPredicateException(this, "precpred(_ctx, 34)");
						setState(608);
						match(T__125);
						}
						break;
					case 40:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(609);
						if (!(precpred(_ctx, 33))) throw new FailedPredicateException(this, "precpred(_ctx, 33)");
						setState(610);
						match(T__126);
						}
						break;
					case 41:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(611);
						if (!(precpred(_ctx, 32))) throw new FailedPredicateException(this, "precpred(_ctx, 32)");
						setState(612);
						match(T__127);
						}
						break;
					case 42:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(613);
						if (!(precpred(_ctx, 31))) throw new FailedPredicateException(this, "precpred(_ctx, 31)");
						setState(614);
						match(T__128);
						}
						break;
					case 43:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(615);
						if (!(precpred(_ctx, 30))) throw new FailedPredicateException(this, "precpred(_ctx, 30)");
						setState(616);
						match(T__129);
						}
						break;
					case 44:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(617);
						if (!(precpred(_ctx, 29))) throw new FailedPredicateException(this, "precpred(_ctx, 29)");
						setState(618);
						match(T__130);
						}
						break;
					case 45:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(619);
						if (!(precpred(_ctx, 28))) throw new FailedPredicateException(this, "precpred(_ctx, 28)");
						setState(620);
						_la = _input.LA(1);
						if ( !(((((_la - 132)) & ~0x3f) == 0 && ((1L << (_la - 132)) & 7L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						}
						break;
					case 46:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(621);
						if (!(precpred(_ctx, 27))) throw new FailedPredicateException(this, "precpred(_ctx, 27)");
						setState(622);
						_la = _input.LA(1);
						if ( !(_la==T__134 || _la==T__135) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(623);
						match(T__12);
						setState(624);
						expression();
						setState(625);
						match(T__13);
						}
						break;
					case 47:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(627);
						if (!(precpred(_ctx, 26))) throw new FailedPredicateException(this, "precpred(_ctx, 26)");
						setState(628);
						_la = _input.LA(1);
						if ( !(((((_la - 137)) & ~0x3f) == 0 && ((1L << (_la - 137)) & 16383L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(629);
						match(T__12);
						setState(630);
						expression();
						setState(631);
						match(T__13);
						}
						break;
					case 48:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(633);
						if (!(precpred(_ctx, 25))) throw new FailedPredicateException(this, "precpred(_ctx, 25)");
						setState(634);
						_la = _input.LA(1);
						if ( !(((((_la - 151)) & ~0x3f) == 0 && ((1L << (_la - 151)) & 511L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(635);
						match(T__12);
						setState(636);
						expression();
						setState(637);
						match(T__13);
						}
						break;
					case 49:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(639);
						if (!(precpred(_ctx, 24))) throw new FailedPredicateException(this, "precpred(_ctx, 24)");
						setState(640);
						_la = _input.LA(1);
						if ( !(((((_la - 160)) & ~0x3f) == 0 && ((1L << (_la - 160)) & 15L) != 0)) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(641);
						match(T__12);
						setState(642);
						expression();
						setState(643);
						match(T__13);
						}
						break;
					case 50:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(645);
						if (!(precpred(_ctx, 23))) throw new FailedPredicateException(this, "precpred(_ctx, 23)");
						setState(646);
						match(T__163);
						setState(647);
						match(T__12);
						setState(648);
						identOptType();
						setState(649);
						match(T__164);
						setState(650);
						expression();
						setState(651);
						match(T__13);
						}
						break;
					case 51:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(653);
						if (!(precpred(_ctx, 22))) throw new FailedPredicateException(this, "precpred(_ctx, 22)");
						setState(654);
						match(T__165);
						setState(655);
						match(T__12);
						setState(656);
						identOptType();
						setState(657);
						match(T__164);
						setState(658);
						expression();
						setState(659);
						match(T__13);
						}
						break;
					case 52:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(661);
						if (!(precpred(_ctx, 21))) throw new FailedPredicateException(this, "precpred(_ctx, 21)");
						setState(662);
						match(T__166);
						setState(663);
						match(T__12);
						setState(664);
						identOptType();
						setState(665);
						match(T__164);
						setState(666);
						expression();
						setState(667);
						match(T__13);
						}
						break;
					case 53:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(669);
						if (!(precpred(_ctx, 20))) throw new FailedPredicateException(this, "precpred(_ctx, 20)");
						setState(670);
						match(T__167);
						setState(671);
						match(T__12);
						setState(672);
						identOptType();
						setState(673);
						match(T__164);
						setState(674);
						expression();
						setState(675);
						match(T__13);
						}
						break;
					case 54:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(677);
						if (!(precpred(_ctx, 19))) throw new FailedPredicateException(this, "precpred(_ctx, 19)");
						setState(678);
						match(T__168);
						setState(679);
						match(T__12);
						setState(680);
						identOptType();
						setState(681);
						match(T__164);
						setState(682);
						expression();
						setState(683);
						match(T__13);
						}
						break;
					case 55:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(685);
						if (!(precpred(_ctx, 18))) throw new FailedPredicateException(this, "precpred(_ctx, 18)");
						setState(686);
						match(T__169);
						setState(687);
						match(T__12);
						setState(688);
						identOptType();
						setState(689);
						match(T__164);
						setState(690);
						expression();
						setState(691);
						match(T__13);
						}
						break;
					case 56:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(693);
						if (!(precpred(_ctx, 17))) throw new FailedPredicateException(this, "precpred(_ctx, 17)");
						setState(694);
						match(T__170);
						setState(695);
						match(T__12);
						setState(696);
						identOptType();
						setState(697);
						match(T__164);
						setState(698);
						expression();
						setState(699);
						match(T__13);
						}
						break;
					case 57:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(701);
						if (!(precpred(_ctx, 16))) throw new FailedPredicateException(this, "precpred(_ctx, 16)");
						setState(702);
						match(T__171);
						setState(703);
						match(T__12);
						setState(704);
						identOptType();
						setState(705);
						match(T__164);
						setState(706);
						expression();
						setState(707);
						match(T__13);
						}
						break;
					case 58:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(709);
						if (!(precpred(_ctx, 15))) throw new FailedPredicateException(this, "precpred(_ctx, 15)");
						setState(710);
						match(T__172);
						setState(711);
						match(T__12);
						setState(712);
						identOptType();
						setState(713);
						match(T__164);
						setState(714);
						expression();
						setState(715);
						match(T__13);
						}
						break;
					case 59:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(717);
						if (!(precpred(_ctx, 14))) throw new FailedPredicateException(this, "precpred(_ctx, 14)");
						setState(718);
						match(T__173);
						setState(719);
						match(T__12);
						setState(720);
						identOptType();
						setState(721);
						match(T__164);
						setState(722);
						expression();
						setState(723);
						match(T__13);
						}
						break;
					case 60:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(725);
						if (!(precpred(_ctx, 13))) throw new FailedPredicateException(this, "precpred(_ctx, 13)");
						setState(726);
						match(T__173);
						setState(727);
						match(T__12);
						setState(728);
						identifier();
						setState(729);
						match(T__13);
						}
						break;
					case 61:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(731);
						if (!(precpred(_ctx, 12))) throw new FailedPredicateException(this, "precpred(_ctx, 12)");
						setState(732);
						match(T__174);
						setState(733);
						match(T__12);
						setState(734);
						identOptType();
						setState(735);
						match(T__164);
						setState(736);
						expression();
						setState(737);
						match(T__13);
						}
						break;
					case 62:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(739);
						if (!(precpred(_ctx, 11))) throw new FailedPredicateException(this, "precpred(_ctx, 11)");
						setState(740);
						match(T__175);
						setState(741);
						match(T__12);
						setState(742);
						expression();
						setState(743);
						match(T__14);
						setState(744);
						expression();
						setState(745);
						match(T__13);
						}
						break;
					case 63:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(747);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(748);
						match(T__176);
						setState(749);
						match(T__12);
						setState(750);
						expression();
						setState(751);
						match(T__14);
						setState(752);
						expression();
						setState(753);
						match(T__13);
						}
						break;
					case 64:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(755);
						if (!(precpred(_ctx, 9))) throw new FailedPredicateException(this, "precpred(_ctx, 9)");
						setState(756);
						match(T__177);
						setState(757);
						match(T__12);
						setState(758);
						expression();
						setState(759);
						match(T__14);
						setState(760);
						expression();
						setState(761);
						match(T__13);
						}
						break;
					case 65:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(763);
						if (!(precpred(_ctx, 8))) throw new FailedPredicateException(this, "precpred(_ctx, 8)");
						setState(764);
						match(T__178);
						setState(765);
						match(T__12);
						setState(766);
						expression();
						setState(767);
						match(T__14);
						setState(768);
						expression();
						setState(769);
						match(T__13);
						}
						break;
					case 66:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(771);
						if (!(precpred(_ctx, 7))) throw new FailedPredicateException(this, "precpred(_ctx, 7)");
						setState(772);
						match(T__179);
						setState(773);
						match(T__12);
						setState(774);
						expression();
						setState(775);
						match(T__14);
						setState(776);
						expression();
						setState(777);
						match(T__13);
						}
						break;
					case 67:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(779);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(780);
						match(T__180);
						setState(781);
						match(T__12);
						setState(782);
						expression();
						setState(783);
						match(T__14);
						setState(784);
						expression();
						setState(785);
						match(T__13);
						}
						break;
					case 68:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(787);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(788);
						match(T__181);
						setState(789);
						match(T__12);
						setState(790);
						expression();
						setState(791);
						match(T__14);
						setState(792);
						expression();
						setState(793);
						match(T__13);
						}
						break;
					case 69:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(795);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(796);
						match(T__182);
						setState(797);
						match(T__12);
						setState(798);
						expression();
						setState(799);
						match(T__14);
						setState(800);
						expression();
						setState(801);
						match(T__13);
						}
						break;
					case 70:
						{
						_localctx = new Factor2ExpressionContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_factor2Expression);
						setState(803);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(804);
						match(T__183);
						setState(805);
						match(T__12);
						setState(806);
						identifier();
						setState(807);
						match(T__9);
						setState(808);
						identifier();
						setState(809);
						match(T__16);
						setState(810);
						expression();
						setState(811);
						match(T__164);
						setState(812);
						expression();
						setState(813);
						match(T__13);
						}
						break;
					}
					} 
				}
				setState(819);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,51,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class IdentOptTypeContext extends ParserRuleContext {
		public TerminalNode ID() { return getToken(SimpleClassModelWithConstraintsParser.ID, 0); }
		public OcltypeContext ocltype() {
			return getRuleContext(OcltypeContext.class,0);
		}
		public IdentOptTypeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_identOptType; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterIdentOptType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitIdentOptType(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitIdentOptType(this);
			else return visitor.visitChildren(this);
		}
	}

	public final IdentOptTypeContext identOptType() throws RecognitionException {
		IdentOptTypeContext _localctx = new IdentOptTypeContext(_ctx, getState());
		enterRule(_localctx, 88, RULE_identOptType);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(820);
			match(ID);
			setState(823);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__8) {
				{
				setState(821);
				match(T__8);
				setState(822);
				ocltype();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class SetExpressionContext extends ParserRuleContext {
		public ExpressionListContext expressionList() {
			return getRuleContext(ExpressionListContext.class,0);
		}
		public SetExpressionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_setExpression; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterSetExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitSetExpression(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitSetExpression(this);
			else return visitor.visitChildren(this);
		}
	}

	public final SetExpressionContext setExpression() throws RecognitionException {
		SetExpressionContext _localctx = new SetExpressionContext(_ctx, getState());
		enterRule(_localctx, 90, RULE_setExpression);
		int _la;
		try {
			setState(850);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__184:
				enterOuterAlt(_localctx, 1);
				{
				setState(825);
				match(T__184);
				setState(827);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7530018577366130688L) != 0) || _la==T__79 || _la==T__80 || ((((_la - 185)) & ~0x3f) == 0 && ((1L << (_la - 185)) & 4095L) != 0)) {
					{
					setState(826);
					expressionList();
					}
				}

				setState(829);
				match(T__4);
				}
				break;
			case T__185:
				enterOuterAlt(_localctx, 2);
				{
				setState(830);
				match(T__185);
				setState(832);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7530018577366130688L) != 0) || _la==T__79 || _la==T__80 || ((((_la - 185)) & ~0x3f) == 0 && ((1L << (_la - 185)) & 4095L) != 0)) {
					{
					setState(831);
					expressionList();
					}
				}

				setState(834);
				match(T__4);
				}
				break;
			case T__186:
				enterOuterAlt(_localctx, 3);
				{
				setState(835);
				match(T__186);
				setState(837);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7530018577366130688L) != 0) || _la==T__79 || _la==T__80 || ((((_la - 185)) & ~0x3f) == 0 && ((1L << (_la - 185)) & 4095L) != 0)) {
					{
					setState(836);
					expressionList();
					}
				}

				setState(839);
				match(T__4);
				}
				break;
			case T__187:
				enterOuterAlt(_localctx, 4);
				{
				setState(840);
				match(T__187);
				setState(842);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7530018577366130688L) != 0) || _la==T__79 || _la==T__80 || ((((_la - 185)) & ~0x3f) == 0 && ((1L << (_la - 185)) & 4095L) != 0)) {
					{
					setState(841);
					expressionList();
					}
				}

				setState(844);
				match(T__4);
				}
				break;
			case T__188:
				enterOuterAlt(_localctx, 5);
				{
				setState(845);
				match(T__188);
				setState(847);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 7530018577366130688L) != 0) || _la==T__79 || _la==T__80 || ((((_la - 185)) & ~0x3f) == 0 && ((1L << (_la - 185)) & 4095L) != 0)) {
					{
					setState(846);
					expressionList();
					}
				}

				setState(849);
				match(T__4);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Qualified_nameContext extends ParserRuleContext {
		public TerminalNode ENUMERATION_LITERAL() { return getToken(SimpleClassModelWithConstraintsParser.ENUMERATION_LITERAL, 0); }
		public Qualified_nameContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_qualified_name; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).enterQualified_name(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof SimpleClassModelWithConstraintsListener ) ((SimpleClassModelWithConstraintsListener)listener).exitQualified_name(this);
		}
		@Override
		public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
			if ( visitor instanceof SimpleClassModelWithConstraintsVisitor ) return ((SimpleClassModelWithConstraintsVisitor<? extends T>)visitor).visitQualified_name(this);
			else return visitor.visitChildren(this);
		}
	}

	public final Qualified_nameContext qualified_name() throws RecognitionException {
		Qualified_nameContext _localctx = new Qualified_nameContext(_ctx, getState());
		enterRule(_localctx, 92, RULE_qualified_name);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(852);
			match(ENUMERATION_LITERAL);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 35:
			return basicExpression_sempred((BasicExpressionContext)_localctx, predIndex);
		case 39:
			return logicalExpression_sempred((LogicalExpressionContext)_localctx, predIndex);
		case 41:
			return additiveExpression_sempred((AdditiveExpressionContext)_localctx, predIndex);
		case 43:
			return factor2Expression_sempred((Factor2ExpressionContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean basicExpression_sempred(BasicExpressionContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 11);
		case 1:
			return precpred(_ctx, 10);
		case 2:
			return precpred(_ctx, 9);
		}
		return true;
	}
	private boolean logicalExpression_sempred(LogicalExpressionContext _localctx, int predIndex) {
		switch (predIndex) {
		case 3:
			return precpred(_ctx, 7);
		case 4:
			return precpred(_ctx, 6);
		case 5:
			return precpred(_ctx, 5);
		case 6:
			return precpred(_ctx, 4);
		case 7:
			return precpred(_ctx, 3);
		case 8:
			return precpred(_ctx, 2);
		}
		return true;
	}
	private boolean additiveExpression_sempred(AdditiveExpressionContext _localctx, int predIndex) {
		switch (predIndex) {
		case 9:
			return precpred(_ctx, 3);
		case 10:
			return precpred(_ctx, 2);
		}
		return true;
	}
	private boolean factor2Expression_sempred(Factor2ExpressionContext _localctx, int predIndex) {
		switch (predIndex) {
		case 11:
			return precpred(_ctx, 72);
		case 12:
			return precpred(_ctx, 71);
		case 13:
			return precpred(_ctx, 70);
		case 14:
			return precpred(_ctx, 69);
		case 15:
			return precpred(_ctx, 68);
		case 16:
			return precpred(_ctx, 67);
		case 17:
			return precpred(_ctx, 66);
		case 18:
			return precpred(_ctx, 65);
		case 19:
			return precpred(_ctx, 64);
		case 20:
			return precpred(_ctx, 63);
		case 21:
			return precpred(_ctx, 62);
		case 22:
			return precpred(_ctx, 61);
		case 23:
			return precpred(_ctx, 60);
		case 24:
			return precpred(_ctx, 59);
		case 25:
			return precpred(_ctx, 58);
		case 26:
			return precpred(_ctx, 57);
		case 27:
			return precpred(_ctx, 56);
		case 28:
			return precpred(_ctx, 55);
		case 29:
			return precpred(_ctx, 54);
		case 30:
			return precpred(_ctx, 53);
		case 31:
			return precpred(_ctx, 52);
		case 32:
			return precpred(_ctx, 51);
		case 33:
			return precpred(_ctx, 50);
		case 34:
			return precpred(_ctx, 49);
		case 35:
			return precpred(_ctx, 48);
		case 36:
			return precpred(_ctx, 47);
		case 37:
			return precpred(_ctx, 46);
		case 38:
			return precpred(_ctx, 45);
		case 39:
			return precpred(_ctx, 44);
		case 40:
			return precpred(_ctx, 43);
		case 41:
			return precpred(_ctx, 42);
		case 42:
			return precpred(_ctx, 41);
		case 43:
			return precpred(_ctx, 40);
		case 44:
			return precpred(_ctx, 39);
		case 45:
			return precpred(_ctx, 38);
		case 46:
			return precpred(_ctx, 37);
		case 47:
			return precpred(_ctx, 36);
		case 48:
			return precpred(_ctx, 35);
		case 49:
			return precpred(_ctx, 34);
		case 50:
			return precpred(_ctx, 33);
		case 51:
			return precpred(_ctx, 32);
		case 52:
			return precpred(_ctx, 31);
		case 53:
			return precpred(_ctx, 30);
		case 54:
			return precpred(_ctx, 29);
		case 55:
			return precpred(_ctx, 28);
		case 56:
			return precpred(_ctx, 27);
		case 57:
			return precpred(_ctx, 26);
		case 58:
			return precpred(_ctx, 25);
		case 59:
			return precpred(_ctx, 24);
		case 60:
			return precpred(_ctx, 23);
		case 61:
			return precpred(_ctx, 22);
		case 62:
			return precpred(_ctx, 21);
		case 63:
			return precpred(_ctx, 20);
		case 64:
			return precpred(_ctx, 19);
		case 65:
			return precpred(_ctx, 18);
		case 66:
			return precpred(_ctx, 17);
		case 67:
			return precpred(_ctx, 16);
		case 68:
			return precpred(_ctx, 15);
		case 69:
			return precpred(_ctx, 14);
		case 70:
			return precpred(_ctx, 13);
		case 71:
			return precpred(_ctx, 12);
		case 72:
			return precpred(_ctx, 11);
		case 73:
			return precpred(_ctx, 10);
		case 74:
			return precpred(_ctx, 9);
		case 75:
			return precpred(_ctx, 8);
		case 76:
			return precpred(_ctx, 7);
		case 77:
			return precpred(_ctx, 6);
		case 78:
			return precpred(_ctx, 5);
		case 79:
			return precpred(_ctx, 4);
		case 80:
			return precpred(_ctx, 3);
		}
		return true;
	}

	public static final String _serializedATN =
		"\u0004\u0001\u00c7\u0357\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001"+
		"\u0002\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004\u0007\u0004"+
		"\u0002\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007\u0007\u0007"+
		"\u0002\b\u0007\b\u0002\t\u0007\t\u0002\n\u0007\n\u0002\u000b\u0007\u000b"+
		"\u0002\f\u0007\f\u0002\r\u0007\r\u0002\u000e\u0007\u000e\u0002\u000f\u0007"+
		"\u000f\u0002\u0010\u0007\u0010\u0002\u0011\u0007\u0011\u0002\u0012\u0007"+
		"\u0012\u0002\u0013\u0007\u0013\u0002\u0014\u0007\u0014\u0002\u0015\u0007"+
		"\u0015\u0002\u0016\u0007\u0016\u0002\u0017\u0007\u0017\u0002\u0018\u0007"+
		"\u0018\u0002\u0019\u0007\u0019\u0002\u001a\u0007\u001a\u0002\u001b\u0007"+
		"\u001b\u0002\u001c\u0007\u001c\u0002\u001d\u0007\u001d\u0002\u001e\u0007"+
		"\u001e\u0002\u001f\u0007\u001f\u0002 \u0007 \u0002!\u0007!\u0002\"\u0007"+
		"\"\u0002#\u0007#\u0002$\u0007$\u0002%\u0007%\u0002&\u0007&\u0002\'\u0007"+
		"\'\u0002(\u0007(\u0002)\u0007)\u0002*\u0007*\u0002+\u0007+\u0002,\u0007"+
		",\u0002-\u0007-\u0002.\u0007.\u0001\u0000\u0001\u0000\u0001\u0000\u0001"+
		"\u0001\u0001\u0001\u0005\u0001d\b\u0001\n\u0001\f\u0001g\t\u0001\u0001"+
		"\u0002\u0001\u0002\u0001\u0002\u0003\u0002l\b\u0002\u0001\u0003\u0001"+
		"\u0003\u0001\u0003\u0001\u0003\u0003\u0003r\b\u0003\u0001\u0003\u0001"+
		"\u0003\u0003\u0003v\b\u0003\u0001\u0003\u0001\u0003\u0003\u0003z\b\u0003"+
		"\u0001\u0003\u0001\u0003\u0001\u0004\u0004\u0004\u007f\b\u0004\u000b\u0004"+
		"\f\u0004\u0080\u0001\u0005\u0001\u0005\u0003\u0005\u0085\b\u0005\u0001"+
		"\u0006\u0001\u0006\u0001\u0006\u0003\u0006\u008a\b\u0006\u0001\u0006\u0001"+
		"\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0001"+
		"\u0006\u0001\u0006\u0001\u0006\u0001\u0006\u0003\u0006\u0097\b\u0006\u0001"+
		"\u0007\u0003\u0007\u009a\b\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0001"+
		"\u0007\u0003\u0007\u00a0\b\u0007\u0001\u0007\u0001\u0007\u0001\u0007\u0001"+
		"\u0007\u0001\u0007\u0001\b\u0001\b\u0001\b\u0005\b\u00aa\b\b\n\b\f\b\u00ad"+
		"\t\b\u0001\b\u0001\b\u0001\t\u0001\t\u0001\t\u0001\t\u0001\n\u0001\n\u0001"+
		"\n\u0005\n\u00b8\b\n\n\n\f\n\u00bb\t\n\u0001\n\u0001\n\u0001\u000b\u0001"+
		"\u000b\u0001\u000b\u0001\u000b\u0001\u000b\u0005\u000b\u00c4\b\u000b\n"+
		"\u000b\f\u000b\u00c7\t\u000b\u0001\u000b\u0001\u000b\u0001\f\u0001\f\u0001"+
		"\f\u0001\f\u0001\f\u0001\f\u0001\f\u0003\f\u00d2\b\f\u0001\r\u0001\r\u0001"+
		"\r\u0001\r\u0004\r\u00d8\b\r\u000b\r\f\r\u00d9\u0001\r\u0001\r\u0001\u000e"+
		"\u0001\u000e\u0001\u000e\u0001\u000e\u0001\u000f\u0001\u000f\u0001\u000f"+
		"\u0003\u000f\u00e5\b\u000f\u0001\u000f\u0001\u000f\u0001\u000f\u0003\u000f"+
		"\u00ea\b\u000f\u0001\u000f\u0001\u000f\u0001\u0010\u0001\u0010\u0001\u0010"+
		"\u0001\u0010\u0001\u0010\u0001\u0011\u0001\u0011\u0001\u0011\u0001\u0011"+
		"\u0001\u0012\u0001\u0012\u0001\u0012\u0001\u0012\u0003\u0012\u00fb\b\u0012"+
		"\u0001\u0012\u0003\u0012\u00fe\b\u0012\u0001\u0012\u0001\u0012\u0003\u0012"+
		"\u0102\b\u0012\u0001\u0012\u0003\u0012\u0105\b\u0012\u0001\u0013\u0001"+
		"\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001\u0013\u0001"+
		"\u0013\u0001\u0013\u0001\u0013\u0003\u0013\u0111\b\u0013\u0001\u0014\u0001"+
		"\u0014\u0004\u0014\u0115\b\u0014\u000b\u0014\f\u0014\u0116\u0001\u0014"+
		"\u0001\u0014\u0001\u0015\u0001\u0015\u0001\u0016\u0001\u0016\u0003\u0016"+
		"\u011f\b\u0016\u0001\u0017\u0001\u0017\u0001\u0018\u0001\u0018\u0004\u0018"+
		"\u0125\b\u0018\u000b\u0018\f\u0018\u0126\u0001\u0019\u0001\u0019\u0001"+
		"\u0019\u0004\u0019\u012c\b\u0019\u000b\u0019\f\u0019\u012d\u0001\u001a"+
		"\u0001\u001a\u0003\u001a\u0132\b\u001a\u0001\u001a\u0001\u001a\u0001\u001a"+
		"\u0001\u001b\u0001\u001b\u0001\u001b\u0001\u001b\u0001\u001b\u0001\u001b"+
		"\u0001\u001c\u0001\u001c\u0001\u001c\u0003\u001c\u0140\b\u001c\u0001\u001c"+
		"\u0001\u001c\u0001\u001c\u0001\u001c\u0001\u001d\u0005\u001d\u0147\b\u001d"+
		"\n\u001d\f\u001d\u014a\t\u001d\u0001\u001d\u0005\u001d\u014d\b\u001d\n"+
		"\u001d\f\u001d\u0150\t\u001d\u0001\u001e\u0001\u001e\u0001\u001e\u0001"+
		"\u001f\u0001\u001f\u0001\u001f\u0001 \u0001 \u0001 \u0001 \u0001 \u0001"+
		" \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001"+
		" \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001"+
		" \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001 \u0001"+
		" \u0001 \u0001 \u0001 \u0001 \u0003 \u0180\b \u0001!\u0001!\u0001!\u0005"+
		"!\u0185\b!\n!\f!\u0188\t!\u0001!\u0001!\u0001\"\u0001\"\u0001\"\u0001"+
		"\"\u0003\"\u0190\b\"\u0001#\u0001#\u0001#\u0001#\u0001#\u0001#\u0001#"+
		"\u0001#\u0001#\u0001#\u0001#\u0001#\u0001#\u0001#\u0003#\u01a0\b#\u0001"+
		"#\u0001#\u0001#\u0001#\u0001#\u0003#\u01a7\b#\u0001#\u0003#\u01aa\b#\u0001"+
		"#\u0001#\u0001#\u0003#\u01af\b#\u0001#\u0001#\u0001#\u0001#\u0001#\u0001"+
		"#\u0005#\u01b7\b#\n#\f#\u01ba\t#\u0001$\u0001$\u0001$\u0001$\u0001$\u0001"+
		"$\u0001$\u0001$\u0001%\u0001%\u0001%\u0001%\u0001%\u0001%\u0001%\u0001"+
		"&\u0001&\u0001&\u0001&\u0001&\u0001&\u0001&\u0001&\u0001&\u0001\'\u0001"+
		"\'\u0001\'\u0001\'\u0003\'\u01d8\b\'\u0001\'\u0001\'\u0001\'\u0001\'\u0001"+
		"\'\u0001\'\u0001\'\u0001\'\u0001\'\u0001\'\u0001\'\u0001\'\u0001\'\u0001"+
		"\'\u0001\'\u0001\'\u0001\'\u0001\'\u0005\'\u01ec\b\'\n\'\f\'\u01ef\t\'"+
		"\u0001(\u0001(\u0001(\u0001(\u0001(\u0003(\u01f6\b(\u0001)\u0001)\u0001"+
		")\u0001)\u0001)\u0001)\u0001)\u0001)\u0001)\u0005)\u0201\b)\n)\f)\u0204"+
		"\t)\u0001*\u0001*\u0001*\u0001*\u0001*\u0003*\u020b\b*\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0003+\u0212\b+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001"+
		"+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0001+\u0005+\u0330"+
		"\b+\n+\f+\u0333\t+\u0001,\u0001,\u0001,\u0003,\u0338\b,\u0001-\u0001-"+
		"\u0003-\u033c\b-\u0001-\u0001-\u0001-\u0003-\u0341\b-\u0001-\u0001-\u0001"+
		"-\u0003-\u0346\b-\u0001-\u0001-\u0001-\u0003-\u034b\b-\u0001-\u0001-\u0001"+
		"-\u0003-\u0350\b-\u0001-\u0003-\u0353\b-\u0001.\u0001.\u0001.\u0000\u0004"+
		"FNRV/\u0000\u0002\u0004\u0006\b\n\f\u000e\u0010\u0012\u0014\u0016\u0018"+
		"\u001a\u001c\u001e \"$&(*,.02468:<>@BDFHJLNPRTVXZ\\\u0000\u000f\u0001"+
		"\u0000\u0007\b\u0001\u0000\u0019\u001a\u0001\u0000\u001b\u001c\u0001\u0000"+
		"!\"\u0001\u0000#&\u0004\u0000\t\t\u0011\u0011\u0019\u0019EK\u0003\u0000"+
		"  33LL\u0002\u0000\u001d\u001dMO\u0002\u0000\u001b\u001cPQ\u0001\u0000"+
		"TZ\u0001\u0000\u0084\u0086\u0001\u0000\u0087\u0088\u0001\u0000\u0089\u0096"+
		"\u0001\u0000\u0097\u009f\u0001\u0000\u00a0\u00a3\u03c3\u0000^\u0001\u0000"+
		"\u0000\u0000\u0002e\u0001\u0000\u0000\u0000\u0004k\u0001\u0000\u0000\u0000"+
		"\u0006m\u0001\u0000\u0000\u0000\b~\u0001\u0000\u0000\u0000\n\u0084\u0001"+
		"\u0000\u0000\u0000\f\u0096\u0001\u0000\u0000\u0000\u000e\u0099\u0001\u0000"+
		"\u0000\u0000\u0010\u00ab\u0001\u0000\u0000\u0000\u0012\u00b0\u0001\u0000"+
		"\u0000\u0000\u0014\u00b9\u0001\u0000\u0000\u0000\u0016\u00be\u0001\u0000"+
		"\u0000\u0000\u0018\u00d1\u0001\u0000\u0000\u0000\u001a\u00d3\u0001\u0000"+
		"\u0000\u0000\u001c\u00dd\u0001\u0000\u0000\u0000\u001e\u00e1\u0001\u0000"+
		"\u0000\u0000 \u00ed\u0001\u0000\u0000\u0000\"\u00f2\u0001\u0000\u0000"+
		"\u0000$\u00f6\u0001\u0000\u0000\u0000&\u0110\u0001\u0000\u0000\u0000("+
		"\u0112\u0001\u0000\u0000\u0000*\u011a\u0001\u0000\u0000\u0000,\u011e\u0001"+
		"\u0000\u0000\u0000.\u0120\u0001\u0000\u0000\u00000\u0124\u0001\u0000\u0000"+
		"\u00002\u0128\u0001\u0000\u0000\u00004\u012f\u0001\u0000\u0000\u00006"+
		"\u0136\u0001\u0000\u0000\u00008\u013c\u0001\u0000\u0000\u0000:\u0148\u0001"+
		"\u0000\u0000\u0000<\u0151\u0001\u0000\u0000\u0000>\u0154\u0001\u0000\u0000"+
		"\u0000@\u017f\u0001\u0000\u0000\u0000B\u0186\u0001\u0000\u0000\u0000D"+
		"\u018f\u0001\u0000\u0000\u0000F\u019f\u0001\u0000\u0000\u0000H\u01bb\u0001"+
		"\u0000\u0000\u0000J\u01c3\u0001\u0000\u0000\u0000L\u01ca\u0001\u0000\u0000"+
		"\u0000N\u01d7\u0001\u0000\u0000\u0000P\u01f5\u0001\u0000\u0000\u0000R"+
		"\u01f7\u0001\u0000\u0000\u0000T\u020a\u0001\u0000\u0000\u0000V\u0211\u0001"+
		"\u0000\u0000\u0000X\u0334\u0001\u0000\u0000\u0000Z\u0352\u0001\u0000\u0000"+
		"\u0000\\\u0354\u0001\u0000\u0000\u0000^_\u0003\u0002\u0001\u0000_`\u0003"+
		"0\u0018\u0000`\u0001\u0001\u0000\u0000\u0000ad\u0003\u0004\u0002\u0000"+
		"bd\u0003\u001e\u000f\u0000ca\u0001\u0000\u0000\u0000cb\u0001\u0000\u0000"+
		"\u0000dg\u0001\u0000\u0000\u0000ec\u0001\u0000\u0000\u0000ef\u0001\u0000"+
		"\u0000\u0000f\u0003\u0001\u0000\u0000\u0000ge\u0001\u0000\u0000\u0000"+
		"hl\u0003\u0006\u0003\u0000il\u0003\u0016\u000b\u0000jl\u0003\u001a\r\u0000"+
		"kh\u0001\u0000\u0000\u0000ki\u0001\u0000\u0000\u0000kj\u0001\u0000\u0000"+
		"\u0000l\u0005\u0001\u0000\u0000\u0000mn\u0005\u0001\u0000\u0000nq\u0003"+
		"*\u0015\u0000op\u0005\u0002\u0000\u0000pr\u0003*\u0015\u0000qo\u0001\u0000"+
		"\u0000\u0000qr\u0001\u0000\u0000\u0000ru\u0001\u0000\u0000\u0000st\u0005"+
		"\u0003\u0000\u0000tv\u0003\u0014\n\u0000us\u0001\u0000\u0000\u0000uv\u0001"+
		"\u0000\u0000\u0000vw\u0001\u0000\u0000\u0000wy\u0005\u0004\u0000\u0000"+
		"xz\u0003\b\u0004\u0000yx\u0001\u0000\u0000\u0000yz\u0001\u0000\u0000\u0000"+
		"z{\u0001\u0000\u0000\u0000{|\u0005\u0005\u0000\u0000|\u0007\u0001\u0000"+
		"\u0000\u0000}\u007f\u0003\n\u0005\u0000~}\u0001\u0000\u0000\u0000\u007f"+
		"\u0080\u0001\u0000\u0000\u0000\u0080~\u0001\u0000\u0000\u0000\u0080\u0081"+
		"\u0001\u0000\u0000\u0000\u0081\t\u0001\u0000\u0000\u0000\u0082\u0085\u0003"+
		"\f\u0006\u0000\u0083\u0085\u0003\u000e\u0007\u0000\u0084\u0082\u0001\u0000"+
		"\u0000\u0000\u0084\u0083\u0001\u0000\u0000\u0000\u0085\u000b\u0001\u0000"+
		"\u0000\u0000\u0086\u0087\u0005\u0006\u0000\u0000\u0087\u0089\u0003*\u0015"+
		"\u0000\u0088\u008a\u0007\u0000\u0000\u0000\u0089\u0088\u0001\u0000\u0000"+
		"\u0000\u0089\u008a\u0001\u0000\u0000\u0000\u008a\u008b\u0001\u0000\u0000"+
		"\u0000\u008b\u008c\u0005\t\u0000\u0000\u008c\u008d\u0003,\u0016\u0000"+
		"\u008d\u008e\u0005\n\u0000\u0000\u008e\u0097\u0001\u0000\u0000\u0000\u008f"+
		"\u0090\u0005\u000b\u0000\u0000\u0090\u0091\u0005\u0006\u0000\u0000\u0091"+
		"\u0092\u0003*\u0015\u0000\u0092\u0093\u0005\t\u0000\u0000\u0093\u0094"+
		"\u0003,\u0016\u0000\u0094\u0095\u0005\n\u0000\u0000\u0095\u0097\u0001"+
		"\u0000\u0000\u0000\u0096\u0086\u0001\u0000\u0000\u0000\u0096\u008f\u0001"+
		"\u0000\u0000\u0000\u0097\r\u0001\u0000\u0000\u0000\u0098\u009a\u0005\u000b"+
		"\u0000\u0000\u0099\u0098\u0001\u0000\u0000\u0000\u0099\u009a\u0001\u0000"+
		"\u0000\u0000\u009a\u009b\u0001\u0000\u0000\u0000\u009b\u009c\u0005\f\u0000"+
		"\u0000\u009c\u009d\u0003*\u0015\u0000\u009d\u009f\u0005\r\u0000\u0000"+
		"\u009e\u00a0\u0003\u0010\b\u0000\u009f\u009e\u0001\u0000\u0000\u0000\u009f"+
		"\u00a0\u0001\u0000\u0000\u0000\u00a0\u00a1\u0001\u0000\u0000\u0000\u00a1"+
		"\u00a2\u0005\u000e\u0000\u0000\u00a2\u00a3\u0005\t\u0000\u0000\u00a3\u00a4"+
		"\u0003,\u0016\u0000\u00a4\u00a5\u0005\n\u0000\u0000\u00a5\u000f\u0001"+
		"\u0000\u0000\u0000\u00a6\u00a7\u0003\u0012\t\u0000\u00a7\u00a8\u0005\u000f"+
		"\u0000\u0000\u00a8\u00aa\u0001\u0000\u0000\u0000\u00a9\u00a6\u0001\u0000"+
		"\u0000\u0000\u00aa\u00ad\u0001\u0000\u0000\u0000\u00ab\u00a9\u0001\u0000"+
		"\u0000\u0000\u00ab\u00ac\u0001\u0000\u0000\u0000\u00ac\u00ae\u0001\u0000"+
		"\u0000\u0000\u00ad\u00ab\u0001\u0000\u0000\u0000\u00ae\u00af\u0003\u0012"+
		"\t\u0000\u00af\u0011\u0001\u0000\u0000\u0000\u00b0\u00b1\u0003*\u0015"+
		"\u0000\u00b1\u00b2\u0005\t\u0000\u0000\u00b2\u00b3\u0003,\u0016\u0000"+
		"\u00b3\u0013\u0001\u0000\u0000\u0000\u00b4\u00b5\u0003*\u0015\u0000\u00b5"+
		"\u00b6\u0005\u000f\u0000\u0000\u00b6\u00b8\u0001\u0000\u0000\u0000\u00b7"+
		"\u00b4\u0001\u0000\u0000\u0000\u00b8\u00bb\u0001\u0000\u0000\u0000\u00b9"+
		"\u00b7\u0001\u0000\u0000\u0000\u00b9\u00ba\u0001\u0000\u0000\u0000\u00ba"+
		"\u00bc\u0001\u0000\u0000\u0000\u00bb\u00b9\u0001\u0000\u0000\u0000\u00bc"+
		"\u00bd\u0003*\u0015\u0000\u00bd\u0015\u0001\u0000\u0000\u0000\u00be\u00bf"+
		"\u0005\u0010\u0000\u0000\u00bf\u00c0\u0003*\u0015\u0000\u00c0\u00c1\u0005"+
		"\u0011\u0000\u0000\u00c1\u00c5\u0005\u0004\u0000\u0000\u00c2\u00c4\u0003"+
		"\u0018\f\u0000\u00c3\u00c2\u0001\u0000\u0000\u0000\u00c4\u00c7\u0001\u0000"+
		"\u0000\u0000\u00c5\u00c3\u0001\u0000\u0000\u0000\u00c5\u00c6\u0001\u0000"+
		"\u0000\u0000\u00c6\u00c8\u0001\u0000\u0000\u0000\u00c7\u00c5\u0001\u0000"+
		"\u0000\u0000\u00c8\u00c9\u0005\u0005\u0000\u0000\u00c9\u0017\u0001\u0000"+
		"\u0000\u0000\u00ca\u00cb\u0005\u0012\u0000\u0000\u00cb\u00cc\u0003*\u0015"+
		"\u0000\u00cc\u00cd\u0005\t\u0000\u0000\u00cd\u00ce\u0003,\u0016\u0000"+
		"\u00ce\u00cf\u0005\n\u0000\u0000\u00cf\u00d2\u0001\u0000\u0000\u0000\u00d0"+
		"\u00d2\u0003\u000e\u0007\u0000\u00d1\u00ca\u0001\u0000\u0000\u0000\u00d1"+
		"\u00d0\u0001\u0000\u0000\u0000\u00d2\u0019\u0001\u0000\u0000\u0000\u00d3"+
		"\u00d4\u0005\u0013\u0000\u0000\u00d4\u00d5\u0003*\u0015\u0000\u00d5\u00d7"+
		"\u0005\u0004\u0000\u0000\u00d6\u00d8\u0003\u001c\u000e\u0000\u00d7\u00d6"+
		"\u0001\u0000\u0000\u0000\u00d8\u00d9\u0001\u0000\u0000\u0000\u00d9\u00d7"+
		"\u0001\u0000\u0000\u0000\u00d9\u00da\u0001\u0000\u0000\u0000\u00da\u00db"+
		"\u0001\u0000\u0000\u0000\u00db\u00dc\u0005\u0005\u0000\u0000\u00dc\u001b"+
		"\u0001\u0000\u0000\u0000\u00dd\u00de\u0005\u0014\u0000\u0000\u00de\u00df"+
		"\u0003*\u0015\u0000\u00df\u00e0\u0005\n\u0000\u0000\u00e0\u001d\u0001"+
		"\u0000\u0000\u0000\u00e1\u00e2\u0005\u0015\u0000\u0000\u00e2\u00e4\u0005"+
		"\u0004\u0000\u0000\u00e3\u00e5\u0003\"\u0011\u0000\u00e4\u00e3\u0001\u0000"+
		"\u0000\u0000\u00e4\u00e5\u0001\u0000\u0000\u0000\u00e5\u00e6\u0001\u0000"+
		"\u0000\u0000\u00e6\u00e7\u0003$\u0012\u0000\u00e7\u00e9\u0003$\u0012\u0000"+
		"\u00e8\u00ea\u0003 \u0010\u0000\u00e9\u00e8\u0001\u0000\u0000\u0000\u00e9"+
		"\u00ea\u0001\u0000\u0000\u0000\u00ea\u00eb\u0001\u0000\u0000\u0000\u00eb"+
		"\u00ec\u0005\u0005\u0000\u0000\u00ec\u001f\u0001\u0000\u0000\u0000\u00ed"+
		"\u00ee\u0005\u0016\u0000\u0000\u00ee\u00ef\u0005\u0004\u0000\u0000\u00ef"+
		"\u00f0\u0003\b\u0004\u0000\u00f0\u00f1\u0005\u0005\u0000\u0000\u00f1!"+
		"\u0001\u0000\u0000\u0000\u00f2\u00f3\u0005\u0017\u0000\u0000\u00f3\u00f4"+
		"\u0005\u0011\u0000\u0000\u00f4\u00f5\u0003*\u0015\u0000\u00f5#\u0001\u0000"+
		"\u0000\u0000\u00f6\u00f7\u0005\u0018\u0000\u0000\u00f7\u00f8\u0005\u0011"+
		"\u0000\u0000\u00f8\u00fa\u0003*\u0015\u0000\u00f9\u00fb\u0003&\u0013\u0000"+
		"\u00fa\u00f9\u0001\u0000\u0000\u0000\u00fa\u00fb\u0001\u0000\u0000\u0000"+
		"\u00fb\u00fd\u0001\u0000\u0000\u0000\u00fc\u00fe\u0007\u0001\u0000\u0000"+
		"\u00fd\u00fc\u0001\u0000\u0000\u0000\u00fd\u00fe\u0001\u0000\u0000\u0000"+
		"\u00fe\u0101\u0001\u0000\u0000\u0000\u00ff\u0100\u0007\u0002\u0000\u0000"+
		"\u0100\u0102\u0003*\u0015\u0000\u0101\u00ff\u0001\u0000\u0000\u0000\u0101"+
		"\u0102\u0001\u0000\u0000\u0000\u0102\u0104\u0001\u0000\u0000\u0000\u0103"+
		"\u0105\u0003(\u0014\u0000\u0104\u0103\u0001\u0000\u0000\u0000\u0104\u0105"+
		"\u0001\u0000\u0000\u0000\u0105%\u0001\u0000\u0000\u0000\u0106\u0111\u0005"+
		"\u001d\u0000\u0000\u0107\u0111\u0005\u001e\u0000\u0000\u0108\u0111\u0005"+
		"\u001f\u0000\u0000\u0109\u0111\u0005\u00be\u0000\u0000\u010a\u010b\u0005"+
		"\u00be\u0000\u0000\u010b\u010c\u0005 \u0000\u0000\u010c\u0111\u0005\u00be"+
		"\u0000\u0000\u010d\u010e\u0005\u00be\u0000\u0000\u010e\u010f\u0005 \u0000"+
		"\u0000\u010f\u0111\u0005\u001d\u0000\u0000\u0110\u0106\u0001\u0000\u0000"+
		"\u0000\u0110\u0107\u0001\u0000\u0000\u0000\u0110\u0108\u0001\u0000\u0000"+
		"\u0000\u0110\u0109\u0001\u0000\u0000\u0000\u0110\u010a\u0001\u0000\u0000"+
		"\u0000\u0110\u010d\u0001\u0000\u0000\u0000\u0111\'\u0001\u0000\u0000\u0000"+
		"\u0112\u0114\u0005\u0004\u0000\u0000\u0113\u0115\u0007\u0003\u0000\u0000"+
		"\u0114\u0113\u0001\u0000\u0000\u0000\u0115\u0116\u0001\u0000\u0000\u0000"+
		"\u0116\u0114\u0001\u0000\u0000\u0000\u0116\u0117\u0001\u0000\u0000\u0000"+
		"\u0117\u0118\u0001\u0000\u0000\u0000\u0118\u0119\u0005\u0005\u0000\u0000"+
		"\u0119)\u0001\u0000\u0000\u0000\u011a\u011b\u0005\u00c4\u0000\u0000\u011b"+
		"+\u0001\u0000\u0000\u0000\u011c\u011f\u0003.\u0017\u0000\u011d\u011f\u0003"+
		"*\u0015\u0000\u011e\u011c\u0001\u0000\u0000\u0000\u011e\u011d\u0001\u0000"+
		"\u0000\u0000\u011f-\u0001\u0000\u0000\u0000\u0120\u0121\u0007\u0004\u0000"+
		"\u0000\u0121/\u0001\u0000\u0000\u0000\u0122\u0125\u00032\u0019\u0000\u0123"+
		"\u0125\u00036\u001b\u0000\u0124\u0122\u0001\u0000\u0000\u0000\u0124\u0123"+
		"\u0001\u0000\u0000\u0000\u0125\u0126\u0001\u0000\u0000\u0000\u0126\u0124"+
		"\u0001\u0000\u0000\u0000\u0126\u0127\u0001\u0000\u0000\u0000\u01271\u0001"+
		"\u0000\u0000\u0000\u0128\u0129\u0005\'\u0000\u0000\u0129\u012b\u0005\u00c4"+
		"\u0000\u0000\u012a\u012c\u00034\u001a\u0000\u012b\u012a\u0001\u0000\u0000"+
		"\u0000\u012c\u012d\u0001\u0000\u0000\u0000\u012d\u012b\u0001\u0000\u0000"+
		"\u0000\u012d\u012e\u0001\u0000\u0000\u0000\u012e3\u0001\u0000\u0000\u0000"+
		"\u012f\u0131\u0005(\u0000\u0000\u0130\u0132\u0005\u00c4\u0000\u0000\u0131"+
		"\u0130\u0001\u0000\u0000\u0000\u0131\u0132\u0001\u0000\u0000\u0000\u0132"+
		"\u0133\u0001\u0000\u0000\u0000\u0133\u0134\u0005\t\u0000\u0000\u0134\u0135"+
		"\u0003D\"\u0000\u01355\u0001\u0000\u0000\u0000\u0136\u0137\u0005\'\u0000"+
		"\u0000\u0137\u0138\u0005\u00c4\u0000\u0000\u0138\u0139\u0005)\u0000\u0000"+
		"\u0139\u013a\u00038\u001c\u0000\u013a\u013b\u0003:\u001d\u0000\u013b7"+
		"\u0001\u0000\u0000\u0000\u013c\u013d\u0003*\u0015\u0000\u013d\u013f\u0005"+
		"\r\u0000\u0000\u013e\u0140\u0003\u0010\b\u0000\u013f\u013e\u0001\u0000"+
		"\u0000\u0000\u013f\u0140\u0001\u0000\u0000\u0000\u0140\u0141\u0001\u0000"+
		"\u0000\u0000\u0141\u0142\u0005\u000e\u0000\u0000\u0142\u0143\u0005\t\u0000"+
		"\u0000\u0143\u0144\u0003@ \u0000\u01449\u0001\u0000\u0000\u0000\u0145"+
		"\u0147\u0003<\u001e\u0000\u0146\u0145\u0001\u0000\u0000\u0000\u0147\u014a"+
		"\u0001\u0000\u0000\u0000\u0148\u0146\u0001\u0000\u0000\u0000\u0148\u0149"+
		"\u0001\u0000\u0000\u0000\u0149\u014e\u0001\u0000\u0000\u0000\u014a\u0148"+
		"\u0001\u0000\u0000\u0000\u014b\u014d\u0003>\u001f\u0000\u014c\u014b\u0001"+
		"\u0000\u0000\u0000\u014d\u0150\u0001\u0000\u0000\u0000\u014e\u014c\u0001"+
		"\u0000\u0000\u0000\u014e\u014f\u0001\u0000\u0000\u0000\u014f;\u0001\u0000"+
		"\u0000\u0000\u0150\u014e\u0001\u0000\u0000\u0000\u0151\u0152\u0005*\u0000"+
		"\u0000\u0152\u0153\u0003D\"\u0000\u0153=\u0001\u0000\u0000\u0000\u0154"+
		"\u0155\u0005+\u0000\u0000\u0155\u0156\u0003D\"\u0000\u0156?\u0001\u0000"+
		"\u0000\u0000\u0157\u0158\u0005,\u0000\u0000\u0158\u0159\u0005\r\u0000"+
		"\u0000\u0159\u015a\u0003@ \u0000\u015a\u015b\u0005\u000e\u0000\u0000\u015b"+
		"\u0180\u0001\u0000\u0000\u0000\u015c\u015d\u0005-\u0000\u0000\u015d\u015e"+
		"\u0005\r\u0000\u0000\u015e\u015f\u0003@ \u0000\u015f\u0160\u0005\u000e"+
		"\u0000\u0000\u0160\u0180\u0001\u0000\u0000\u0000\u0161\u0162\u0005.\u0000"+
		"\u0000\u0162\u0163\u0005\r\u0000\u0000\u0163\u0164\u0003@ \u0000\u0164"+
		"\u0165\u0005\u000e\u0000\u0000\u0165\u0180\u0001\u0000\u0000\u0000\u0166"+
		"\u0167\u0005/\u0000\u0000\u0167\u0168\u0005\r\u0000\u0000\u0168\u0169"+
		"\u0003@ \u0000\u0169\u016a\u0005\u000e\u0000\u0000\u016a\u0180\u0001\u0000"+
		"\u0000\u0000\u016b\u016c\u00050\u0000\u0000\u016c\u016d\u0005\r\u0000"+
		"\u0000\u016d\u016e\u0003@ \u0000\u016e\u016f\u0005\u000e\u0000\u0000\u016f"+
		"\u0180\u0001\u0000\u0000\u0000\u0170\u0171\u00051\u0000\u0000\u0171\u0172"+
		"\u0005\r\u0000\u0000\u0172\u0173\u0003@ \u0000\u0173\u0174\u0005\u000f"+
		"\u0000\u0000\u0174\u0175\u0003@ \u0000\u0175\u0176\u0005\u000e\u0000\u0000"+
		"\u0176\u0180\u0001\u0000\u0000\u0000\u0177\u0178\u00052\u0000\u0000\u0178"+
		"\u0179\u0005\r\u0000\u0000\u0179\u017a\u0003@ \u0000\u017a\u017b\u0005"+
		"\u000f\u0000\u0000\u017b\u017c\u0003@ \u0000\u017c\u017d\u0005\u000e\u0000"+
		"\u0000\u017d\u0180\u0001\u0000\u0000\u0000\u017e\u0180\u0003,\u0016\u0000"+
		"\u017f\u0157\u0001\u0000\u0000\u0000\u017f\u015c\u0001\u0000\u0000\u0000"+
		"\u017f\u0161\u0001\u0000\u0000\u0000\u017f\u0166\u0001\u0000\u0000\u0000"+
		"\u017f\u016b\u0001\u0000\u0000\u0000\u017f\u0170\u0001\u0000\u0000\u0000"+
		"\u017f\u0177\u0001\u0000\u0000\u0000\u017f\u017e\u0001\u0000\u0000\u0000"+
		"\u0180A\u0001\u0000\u0000\u0000\u0181\u0182\u0003D\"\u0000\u0182\u0183"+
		"\u0005\u000f\u0000\u0000\u0183\u0185\u0001\u0000\u0000\u0000\u0184\u0181"+
		"\u0001\u0000\u0000\u0000\u0185\u0188\u0001\u0000\u0000\u0000\u0186\u0184"+
		"\u0001\u0000\u0000\u0000\u0186\u0187\u0001\u0000\u0000\u0000\u0187\u0189"+
		"\u0001\u0000\u0000\u0000\u0188\u0186\u0001\u0000\u0000\u0000\u0189\u018a"+
		"\u0003D\"\u0000\u018aC\u0001\u0000\u0000\u0000\u018b\u0190\u0003N\'\u0000"+
		"\u018c\u0190\u0003H$\u0000\u018d\u0190\u0003J%\u0000\u018e\u0190\u0003"+
		"L&\u0000\u018f\u018b\u0001\u0000\u0000\u0000\u018f\u018c\u0001\u0000\u0000"+
		"\u0000\u018f\u018d\u0001\u0000\u0000\u0000\u018f\u018e\u0001\u0000\u0000"+
		"\u0000\u0190E\u0001\u0000\u0000\u0000\u0191\u0192\u0006#\uffff\uffff\u0000"+
		"\u0192\u01a0\u0005\u00c3\u0000\u0000\u0193\u0194\u0005\u00c4\u0000\u0000"+
		"\u0194\u01a0\u00056\u0000\u0000\u0195\u01a0\u0005\u00be\u0000\u0000\u0196"+
		"\u01a0\u0005\u00bf\u0000\u0000\u0197\u01a0\u0005\u00c0\u0000\u0000\u0198"+
		"\u01a0\u0005\u00c1\u0000\u0000\u0199\u01a0\u0005\u00c2\u0000\u0000\u019a"+
		"\u01a0\u0005\u00c4\u0000\u0000\u019b\u019c\u0005\r\u0000\u0000\u019c\u019d"+
		"\u0003D\"\u0000\u019d\u019e\u0005\u000e\u0000\u0000\u019e\u01a0\u0001"+
		"\u0000\u0000\u0000\u019f\u0191\u0001\u0000\u0000\u0000\u019f\u0193\u0001"+
		"\u0000\u0000\u0000\u019f\u0195\u0001\u0000\u0000\u0000\u019f\u0196\u0001"+
		"\u0000\u0000\u0000\u019f\u0197\u0001\u0000\u0000\u0000\u019f\u0198\u0001"+
		"\u0000\u0000\u0000\u019f\u0199\u0001\u0000\u0000\u0000\u019f\u019a\u0001"+
		"\u0000\u0000\u0000\u019f\u019b\u0001\u0000\u0000\u0000\u01a0\u01b8\u0001"+
		"\u0000\u0000\u0000\u01a1\u01a2\n\u000b\u0000\u0000\u01a2\u01a3\u00053"+
		"\u0000\u0000\u01a3\u01a9\u0005\u00c4\u0000\u0000\u01a4\u01a6\u0005\r\u0000"+
		"\u0000\u01a5\u01a7\u0003B!\u0000\u01a6\u01a5\u0001\u0000\u0000\u0000\u01a6"+
		"\u01a7\u0001\u0000\u0000\u0000\u01a7\u01a8\u0001\u0000\u0000\u0000\u01a8"+
		"\u01aa\u0005\u000e\u0000\u0000\u01a9\u01a4\u0001\u0000\u0000\u0000\u01a9"+
		"\u01aa\u0001\u0000\u0000\u0000\u01aa\u01b7\u0001\u0000\u0000\u0000\u01ab"+
		"\u01ac\n\n\u0000\u0000\u01ac\u01ae\u0005\r\u0000\u0000\u01ad\u01af\u0003"+
		"B!\u0000\u01ae\u01ad\u0001\u0000\u0000\u0000\u01ae\u01af\u0001\u0000\u0000"+
		"\u0000\u01af\u01b0\u0001\u0000\u0000\u0000\u01b0\u01b7\u0005\u000e\u0000"+
		"\u0000\u01b1\u01b2\n\t\u0000\u0000\u01b2\u01b3\u00054\u0000\u0000\u01b3"+
		"\u01b4\u0003D\"\u0000\u01b4\u01b5\u00055\u0000\u0000\u01b5\u01b7\u0001"+
		"\u0000\u0000\u0000\u01b6\u01a1\u0001\u0000\u0000\u0000\u01b6\u01ab\u0001"+
		"\u0000\u0000\u0000\u01b6\u01b1\u0001\u0000\u0000\u0000\u01b7\u01ba\u0001"+
		"\u0000\u0000\u0000\u01b8\u01b6\u0001\u0000\u0000\u0000\u01b8\u01b9\u0001"+
		"\u0000\u0000\u0000\u01b9G\u0001\u0000\u0000\u0000\u01ba\u01b8\u0001\u0000"+
		"\u0000\u0000\u01bb\u01bc\u00057\u0000\u0000\u01bc\u01bd\u0003D\"\u0000"+
		"\u01bd\u01be\u00058\u0000\u0000\u01be\u01bf\u0003D\"\u0000\u01bf\u01c0"+
		"\u00059\u0000\u0000\u01c0\u01c1\u0003D\"\u0000\u01c1\u01c2\u0005:\u0000"+
		"\u0000\u01c2I\u0001\u0000\u0000\u0000\u01c3\u01c4\u0005;\u0000\u0000\u01c4"+
		"\u01c5\u0003*\u0015\u0000\u01c5\u01c6\u0005\t\u0000\u0000\u01c6\u01c7"+
		"\u0003@ \u0000\u01c7\u01c8\u0005<\u0000\u0000\u01c8\u01c9\u0003D\"\u0000"+
		"\u01c9K\u0001\u0000\u0000\u0000\u01ca\u01cb\u0005=\u0000\u0000\u01cb\u01cc"+
		"\u0005\u00c4\u0000\u0000\u01cc\u01cd\u0005\t\u0000\u0000\u01cd\u01ce\u0003"+
		"@ \u0000\u01ce\u01cf\u0005\u0011\u0000\u0000\u01cf\u01d0\u0003D\"\u0000"+
		"\u01d0\u01d1\u0005<\u0000\u0000\u01d1\u01d2\u0003D\"\u0000\u01d2M\u0001"+
		"\u0000\u0000\u0000\u01d3\u01d4\u0006\'\uffff\uffff\u0000\u01d4\u01d5\u0005"+
		">\u0000\u0000\u01d5\u01d8\u0003N\'\b\u01d6\u01d8\u0003P(\u0000\u01d7\u01d3"+
		"\u0001\u0000\u0000\u0000\u01d7\u01d6\u0001\u0000\u0000\u0000\u01d8\u01ed"+
		"\u0001\u0000\u0000\u0000\u01d9\u01da\n\u0007\u0000\u0000\u01da\u01db\u0005"+
		"?\u0000\u0000\u01db\u01ec\u0003N\'\b\u01dc\u01dd\n\u0006\u0000\u0000\u01dd"+
		"\u01de\u0005@\u0000\u0000\u01de\u01ec\u0003N\'\u0007\u01df\u01e0\n\u0005"+
		"\u0000\u0000\u01e0\u01e1\u0005A\u0000\u0000\u01e1\u01ec\u0003N\'\u0006"+
		"\u01e2\u01e3\n\u0004\u0000\u0000\u01e3\u01e4\u0005B\u0000\u0000\u01e4"+
		"\u01ec\u0003N\'\u0005\u01e5\u01e6\n\u0003\u0000\u0000\u01e6\u01e7\u0005"+
		"C\u0000\u0000\u01e7\u01ec\u0003N\'\u0004\u01e8\u01e9\n\u0002\u0000\u0000"+
		"\u01e9\u01ea\u0005D\u0000\u0000\u01ea\u01ec\u0003N\'\u0003\u01eb\u01d9"+
		"\u0001\u0000\u0000\u0000\u01eb\u01dc\u0001\u0000\u0000\u0000\u01eb\u01df"+
		"\u0001\u0000\u0000\u0000\u01eb\u01e2\u0001\u0000\u0000\u0000\u01eb\u01e5"+
		"\u0001\u0000\u0000\u0000\u01eb\u01e8\u0001\u0000\u0000\u0000\u01ec\u01ef"+
		"\u0001\u0000\u0000\u0000\u01ed\u01eb\u0001\u0000\u0000\u0000\u01ed\u01ee"+
		"\u0001\u0000\u0000\u0000\u01eeO\u0001\u0000\u0000\u0000\u01ef\u01ed\u0001"+
		"\u0000\u0000\u0000\u01f0\u01f1\u0003R)\u0000\u01f1\u01f2\u0007\u0005\u0000"+
		"\u0000\u01f2\u01f3\u0003R)\u0000\u01f3\u01f6\u0001\u0000\u0000\u0000\u01f4"+
		"\u01f6\u0003R)\u0000\u01f5\u01f0\u0001\u0000\u0000\u0000\u01f5\u01f4\u0001"+
		"\u0000\u0000\u0000\u01f6Q\u0001\u0000\u0000\u0000\u01f7\u01f8\u0006)\uffff"+
		"\uffff\u0000\u01f8\u01f9\u0003T*\u0000\u01f9\u0202\u0001\u0000\u0000\u0000"+
		"\u01fa\u01fb\n\u0003\u0000\u0000\u01fb\u01fc\u0007\u0002\u0000\u0000\u01fc"+
		"\u0201\u0003R)\u0004\u01fd\u01fe\n\u0002\u0000\u0000\u01fe\u01ff\u0007"+
		"\u0006\u0000\u0000\u01ff\u0201\u0003R)\u0003\u0200\u01fa\u0001\u0000\u0000"+
		"\u0000\u0200\u01fd\u0001\u0000\u0000\u0000\u0201\u0204\u0001\u0000\u0000"+
		"\u0000\u0202\u0200\u0001\u0000\u0000\u0000\u0202\u0203\u0001\u0000\u0000"+
		"\u0000\u0203S\u0001\u0000\u0000\u0000\u0204\u0202\u0001\u0000\u0000\u0000"+
		"\u0205\u0206\u0003V+\u0000\u0206\u0207\u0007\u0007\u0000\u0000\u0207\u0208"+
		"\u0003T*\u0000\u0208\u020b\u0001\u0000\u0000\u0000\u0209\u020b\u0003V"+
		"+\u0000\u020a\u0205\u0001\u0000\u0000\u0000\u020a\u0209\u0001\u0000\u0000"+
		"\u0000\u020bU\u0001\u0000\u0000\u0000\u020c\u020d\u0006+\uffff\uffff\u0000"+
		"\u020d\u020e\u0007\b\u0000\u0000\u020e\u0212\u0003V+I\u020f\u0212\u0003"+
		"Z-\u0000\u0210\u0212\u0003F#\u0000\u0211\u020c\u0001\u0000\u0000\u0000"+
		"\u0211\u020f\u0001\u0000\u0000\u0000\u0211\u0210\u0001\u0000\u0000\u0000"+
		"\u0212\u0331\u0001\u0000\u0000\u0000\u0213\u0214\nH\u0000\u0000\u0214"+
		"\u0330\u0005R\u0000\u0000\u0215\u0216\nG\u0000\u0000\u0216\u0330\u0005"+
		"S\u0000\u0000\u0217\u0218\nF\u0000\u0000\u0218\u0330\u0007\t\u0000\u0000"+
		"\u0219\u021a\nE\u0000\u0000\u021a\u0330\u0005[\u0000\u0000\u021b\u021c"+
		"\nD\u0000\u0000\u021c\u0330\u0005\\\u0000\u0000\u021d\u021e\nC\u0000\u0000"+
		"\u021e\u0330\u0005]\u0000\u0000\u021f\u0220\nB\u0000\u0000\u0220\u0330"+
		"\u0005^\u0000\u0000\u0221\u0222\nA\u0000\u0000\u0222\u0330\u0005_\u0000"+
		"\u0000\u0223\u0224\n@\u0000\u0000\u0224\u0330\u0005`\u0000\u0000\u0225"+
		"\u0226\n?\u0000\u0000\u0226\u0330\u0005a\u0000\u0000\u0227\u0228\n>\u0000"+
		"\u0000\u0228\u0330\u0005b\u0000\u0000\u0229\u022a\n=\u0000\u0000\u022a"+
		"\u0330\u0005c\u0000\u0000\u022b\u022c\n<\u0000\u0000\u022c\u0330\u0005"+
		"d\u0000\u0000\u022d\u022e\n;\u0000\u0000\u022e\u0330\u0005e\u0000\u0000"+
		"\u022f\u0230\n:\u0000\u0000\u0230\u0330\u0005f\u0000\u0000\u0231\u0232"+
		"\n9\u0000\u0000\u0232\u0330\u0005g\u0000\u0000\u0233\u0234\n8\u0000\u0000"+
		"\u0234\u0330\u0005h\u0000\u0000\u0235\u0236\n7\u0000\u0000\u0236\u0330"+
		"\u0005i\u0000\u0000\u0237\u0238\n6\u0000\u0000\u0238\u0330\u0005j\u0000"+
		"\u0000\u0239\u023a\n5\u0000\u0000\u023a\u0330\u0005k\u0000\u0000\u023b"+
		"\u023c\n4\u0000\u0000\u023c\u0330\u0005l\u0000\u0000\u023d\u023e\n3\u0000"+
		"\u0000\u023e\u0330\u0005m\u0000\u0000\u023f\u0240\n2\u0000\u0000\u0240"+
		"\u0330\u0005n\u0000\u0000\u0241\u0242\n1\u0000\u0000\u0242\u0330\u0005"+
		"o\u0000\u0000\u0243\u0244\n0\u0000\u0000\u0244\u0330\u0005p\u0000\u0000"+
		"\u0245\u0246\n/\u0000\u0000\u0246\u0330\u0005q\u0000\u0000\u0247\u0248"+
		"\n.\u0000\u0000\u0248\u0330\u0005r\u0000\u0000\u0249\u024a\n-\u0000\u0000"+
		"\u024a\u0330\u0005s\u0000\u0000\u024b\u024c\n,\u0000\u0000\u024c\u0330"+
		"\u0005t\u0000\u0000\u024d\u024e\n+\u0000\u0000\u024e\u0330\u0005u\u0000"+
		"\u0000\u024f\u0250\n*\u0000\u0000\u0250\u0330\u0005v\u0000\u0000\u0251"+
		"\u0252\n)\u0000\u0000\u0252\u0330\u0005w\u0000\u0000\u0253\u0254\n(\u0000"+
		"\u0000\u0254\u0330\u0005x\u0000\u0000\u0255\u0256\n\'\u0000\u0000\u0256"+
		"\u0330\u0005y\u0000\u0000\u0257\u0258\n&\u0000\u0000\u0258\u0330\u0005"+
		"z\u0000\u0000\u0259\u025a\n%\u0000\u0000\u025a\u0330\u0005{\u0000\u0000"+
		"\u025b\u025c\n$\u0000\u0000\u025c\u0330\u0005|\u0000\u0000\u025d\u025e"+
		"\n#\u0000\u0000\u025e\u0330\u0005}\u0000\u0000\u025f\u0260\n\"\u0000\u0000"+
		"\u0260\u0330\u0005~\u0000\u0000\u0261\u0262\n!\u0000\u0000\u0262\u0330"+
		"\u0005\u007f\u0000\u0000\u0263\u0264\n \u0000\u0000\u0264\u0330\u0005"+
		"\u0080\u0000\u0000\u0265\u0266\n\u001f\u0000\u0000\u0266\u0330\u0005\u0081"+
		"\u0000\u0000\u0267\u0268\n\u001e\u0000\u0000\u0268\u0330\u0005\u0082\u0000"+
		"\u0000\u0269\u026a\n\u001d\u0000\u0000\u026a\u0330\u0005\u0083\u0000\u0000"+
		"\u026b\u026c\n\u001c\u0000\u0000\u026c\u0330\u0007\n\u0000\u0000\u026d"+
		"\u026e\n\u001b\u0000\u0000\u026e\u026f\u0007\u000b\u0000\u0000\u026f\u0270"+
		"\u0005\r\u0000\u0000\u0270\u0271\u0003D\"\u0000\u0271\u0272\u0005\u000e"+
		"\u0000\u0000\u0272\u0330\u0001\u0000\u0000\u0000\u0273\u0274\n\u001a\u0000"+
		"\u0000\u0274\u0275\u0007\f\u0000\u0000\u0275\u0276\u0005\r\u0000\u0000"+
		"\u0276\u0277\u0003D\"\u0000\u0277\u0278\u0005\u000e\u0000\u0000\u0278"+
		"\u0330\u0001\u0000\u0000\u0000\u0279\u027a\n\u0019\u0000\u0000\u027a\u027b"+
		"\u0007\r\u0000\u0000\u027b\u027c\u0005\r\u0000\u0000\u027c\u027d\u0003"+
		"D\"\u0000\u027d\u027e\u0005\u000e\u0000\u0000\u027e\u0330\u0001\u0000"+
		"\u0000\u0000\u027f\u0280\n\u0018\u0000\u0000\u0280\u0281\u0007\u000e\u0000"+
		"\u0000\u0281\u0282\u0005\r\u0000\u0000\u0282\u0283\u0003D\"\u0000\u0283"+
		"\u0284\u0005\u000e\u0000\u0000\u0284\u0330\u0001\u0000\u0000\u0000\u0285"+
		"\u0286\n\u0017\u0000\u0000\u0286\u0287\u0005\u00a4\u0000\u0000\u0287\u0288"+
		"\u0005\r\u0000\u0000\u0288\u0289\u0003X,\u0000\u0289\u028a\u0005\u00a5"+
		"\u0000\u0000\u028a\u028b\u0003D\"\u0000\u028b\u028c\u0005\u000e\u0000"+
		"\u0000\u028c\u0330\u0001\u0000\u0000\u0000\u028d\u028e\n\u0016\u0000\u0000"+
		"\u028e\u028f\u0005\u00a6\u0000\u0000\u028f\u0290\u0005\r\u0000\u0000\u0290"+
		"\u0291\u0003X,\u0000\u0291\u0292\u0005\u00a5\u0000\u0000\u0292\u0293\u0003"+
		"D\"\u0000\u0293\u0294\u0005\u000e\u0000\u0000\u0294\u0330\u0001\u0000"+
		"\u0000\u0000\u0295\u0296\n\u0015\u0000\u0000\u0296\u0297\u0005\u00a7\u0000"+
		"\u0000\u0297\u0298\u0005\r\u0000\u0000\u0298\u0299\u0003X,\u0000\u0299"+
		"\u029a\u0005\u00a5\u0000\u0000\u029a\u029b\u0003D\"\u0000\u029b\u029c"+
		"\u0005\u000e\u0000\u0000\u029c\u0330\u0001\u0000\u0000\u0000\u029d\u029e"+
		"\n\u0014\u0000\u0000\u029e\u029f\u0005\u00a8\u0000\u0000\u029f\u02a0\u0005"+
		"\r\u0000\u0000\u02a0\u02a1\u0003X,\u0000\u02a1\u02a2\u0005\u00a5\u0000"+
		"\u0000\u02a2\u02a3\u0003D\"\u0000\u02a3\u02a4\u0005\u000e\u0000\u0000"+
		"\u02a4\u0330\u0001\u0000\u0000\u0000\u02a5\u02a6\n\u0013\u0000\u0000\u02a6"+
		"\u02a7\u0005\u00a9\u0000\u0000\u02a7\u02a8\u0005\r\u0000\u0000\u02a8\u02a9"+
		"\u0003X,\u0000\u02a9\u02aa\u0005\u00a5\u0000\u0000\u02aa\u02ab\u0003D"+
		"\"\u0000\u02ab\u02ac\u0005\u000e\u0000\u0000\u02ac\u0330\u0001\u0000\u0000"+
		"\u0000\u02ad\u02ae\n\u0012\u0000\u0000\u02ae\u02af\u0005\u00aa\u0000\u0000"+
		"\u02af\u02b0\u0005\r\u0000\u0000\u02b0\u02b1\u0003X,\u0000\u02b1\u02b2"+
		"\u0005\u00a5\u0000\u0000\u02b2\u02b3\u0003D\"\u0000\u02b3\u02b4\u0005"+
		"\u000e\u0000\u0000\u02b4\u0330\u0001\u0000\u0000\u0000\u02b5\u02b6\n\u0011"+
		"\u0000\u0000\u02b6\u02b7\u0005\u00ab\u0000\u0000\u02b7\u02b8\u0005\r\u0000"+
		"\u0000\u02b8\u02b9\u0003X,\u0000\u02b9\u02ba\u0005\u00a5\u0000\u0000\u02ba"+
		"\u02bb\u0003D\"\u0000\u02bb\u02bc\u0005\u000e\u0000\u0000\u02bc\u0330"+
		"\u0001\u0000\u0000\u0000\u02bd\u02be\n\u0010\u0000\u0000\u02be\u02bf\u0005"+
		"\u00ac\u0000\u0000\u02bf\u02c0\u0005\r\u0000\u0000\u02c0\u02c1\u0003X"+
		",\u0000\u02c1\u02c2\u0005\u00a5\u0000\u0000\u02c2\u02c3\u0003D\"\u0000"+
		"\u02c3\u02c4\u0005\u000e\u0000\u0000\u02c4\u0330\u0001\u0000\u0000\u0000"+
		"\u02c5\u02c6\n\u000f\u0000\u0000\u02c6\u02c7\u0005\u00ad\u0000\u0000\u02c7"+
		"\u02c8\u0005\r\u0000\u0000\u02c8\u02c9\u0003X,\u0000\u02c9\u02ca\u0005"+
		"\u00a5\u0000\u0000\u02ca\u02cb\u0003D\"\u0000\u02cb\u02cc\u0005\u000e"+
		"\u0000\u0000\u02cc\u0330\u0001\u0000\u0000\u0000\u02cd\u02ce\n\u000e\u0000"+
		"\u0000\u02ce\u02cf\u0005\u00ae\u0000\u0000\u02cf\u02d0\u0005\r\u0000\u0000"+
		"\u02d0\u02d1\u0003X,\u0000\u02d1\u02d2\u0005\u00a5\u0000\u0000\u02d2\u02d3"+
		"\u0003D\"\u0000\u02d3\u02d4\u0005\u000e\u0000\u0000\u02d4\u0330\u0001"+
		"\u0000\u0000\u0000\u02d5\u02d6\n\r\u0000\u0000\u02d6\u02d7\u0005\u00ae"+
		"\u0000\u0000\u02d7\u02d8\u0005\r\u0000\u0000\u02d8\u02d9\u0003*\u0015"+
		"\u0000\u02d9\u02da\u0005\u000e\u0000\u0000\u02da\u0330\u0001\u0000\u0000"+
		"\u0000\u02db\u02dc\n\f\u0000\u0000\u02dc\u02dd\u0005\u00af\u0000\u0000"+
		"\u02dd\u02de\u0005\r\u0000\u0000\u02de\u02df\u0003X,\u0000\u02df\u02e0"+
		"\u0005\u00a5\u0000\u0000\u02e0\u02e1\u0003D\"\u0000\u02e1\u02e2\u0005"+
		"\u000e\u0000\u0000\u02e2\u0330\u0001\u0000\u0000\u0000\u02e3\u02e4\n\u000b"+
		"\u0000\u0000\u02e4\u02e5\u0005\u00b0\u0000\u0000\u02e5\u02e6\u0005\r\u0000"+
		"\u0000\u02e6\u02e7\u0003D\"\u0000\u02e7\u02e8\u0005\u000f\u0000\u0000"+
		"\u02e8\u02e9\u0003D\"\u0000\u02e9\u02ea\u0005\u000e\u0000\u0000\u02ea"+
		"\u0330\u0001\u0000\u0000\u0000\u02eb\u02ec\n\n\u0000\u0000\u02ec\u02ed"+
		"\u0005\u00b1\u0000\u0000\u02ed\u02ee\u0005\r\u0000\u0000\u02ee\u02ef\u0003"+
		"D\"\u0000\u02ef\u02f0\u0005\u000f\u0000\u0000\u02f0\u02f1\u0003D\"\u0000"+
		"\u02f1\u02f2\u0005\u000e\u0000\u0000\u02f2\u0330\u0001\u0000\u0000\u0000"+
		"\u02f3\u02f4\n\t\u0000\u0000\u02f4\u02f5\u0005\u00b2\u0000\u0000\u02f5"+
		"\u02f6\u0005\r\u0000\u0000\u02f6\u02f7\u0003D\"\u0000\u02f7\u02f8\u0005"+
		"\u000f\u0000\u0000\u02f8\u02f9\u0003D\"\u0000\u02f9\u02fa\u0005\u000e"+
		"\u0000\u0000\u02fa\u0330\u0001\u0000\u0000\u0000\u02fb\u02fc\n\b\u0000"+
		"\u0000\u02fc\u02fd\u0005\u00b3\u0000\u0000\u02fd\u02fe\u0005\r\u0000\u0000"+
		"\u02fe\u02ff\u0003D\"\u0000\u02ff\u0300\u0005\u000f\u0000\u0000\u0300"+
		"\u0301\u0003D\"\u0000\u0301\u0302\u0005\u000e\u0000\u0000\u0302\u0330"+
		"\u0001\u0000\u0000\u0000\u0303\u0304\n\u0007\u0000\u0000\u0304\u0305\u0005"+
		"\u00b4\u0000\u0000\u0305\u0306\u0005\r\u0000\u0000\u0306\u0307\u0003D"+
		"\"\u0000\u0307\u0308\u0005\u000f\u0000\u0000\u0308\u0309\u0003D\"\u0000"+
		"\u0309\u030a\u0005\u000e\u0000\u0000\u030a\u0330\u0001\u0000\u0000\u0000"+
		"\u030b\u030c\n\u0006\u0000\u0000\u030c\u030d\u0005\u00b5\u0000\u0000\u030d"+
		"\u030e\u0005\r\u0000\u0000\u030e\u030f\u0003D\"\u0000\u030f\u0310\u0005"+
		"\u000f\u0000\u0000\u0310\u0311\u0003D\"\u0000\u0311\u0312\u0005\u000e"+
		"\u0000\u0000\u0312\u0330\u0001\u0000\u0000\u0000\u0313\u0314\n\u0005\u0000"+
		"\u0000\u0314\u0315\u0005\u00b6\u0000\u0000\u0315\u0316\u0005\r\u0000\u0000"+
		"\u0316\u0317\u0003D\"\u0000\u0317\u0318\u0005\u000f\u0000\u0000\u0318"+
		"\u0319\u0003D\"\u0000\u0319\u031a\u0005\u000e\u0000\u0000\u031a\u0330"+
		"\u0001\u0000\u0000\u0000\u031b\u031c\n\u0004\u0000\u0000\u031c\u031d\u0005"+
		"\u00b7\u0000\u0000\u031d\u031e\u0005\r\u0000\u0000\u031e\u031f\u0003D"+
		"\"\u0000\u031f\u0320\u0005\u000f\u0000\u0000\u0320\u0321\u0003D\"\u0000"+
		"\u0321\u0322\u0005\u000e\u0000\u0000\u0322\u0330\u0001\u0000\u0000\u0000"+
		"\u0323\u0324\n\u0003\u0000\u0000\u0324\u0325\u0005\u00b8\u0000\u0000\u0325"+
		"\u0326\u0005\r\u0000\u0000\u0326\u0327\u0003*\u0015\u0000\u0327\u0328"+
		"\u0005\n\u0000\u0000\u0328\u0329\u0003*\u0015\u0000\u0329\u032a\u0005"+
		"\u0011\u0000\u0000\u032a\u032b\u0003D\"\u0000\u032b\u032c\u0005\u00a5"+
		"\u0000\u0000\u032c\u032d\u0003D\"\u0000\u032d\u032e\u0005\u000e\u0000"+
		"\u0000\u032e\u0330\u0001\u0000\u0000\u0000\u032f\u0213\u0001\u0000\u0000"+
		"\u0000\u032f\u0215\u0001\u0000\u0000\u0000\u032f\u0217\u0001\u0000\u0000"+
		"\u0000\u032f\u0219\u0001\u0000\u0000\u0000\u032f\u021b\u0001\u0000\u0000"+
		"\u0000\u032f\u021d\u0001\u0000\u0000\u0000\u032f\u021f\u0001\u0000\u0000"+
		"\u0000\u032f\u0221\u0001\u0000\u0000\u0000\u032f\u0223\u0001\u0000\u0000"+
		"\u0000\u032f\u0225\u0001\u0000\u0000\u0000\u032f\u0227\u0001\u0000\u0000"+
		"\u0000\u032f\u0229\u0001\u0000\u0000\u0000\u032f\u022b\u0001\u0000\u0000"+
		"\u0000\u032f\u022d\u0001\u0000\u0000\u0000\u032f\u022f\u0001\u0000\u0000"+
		"\u0000\u032f\u0231\u0001\u0000\u0000\u0000\u032f\u0233\u0001\u0000\u0000"+
		"\u0000\u032f\u0235\u0001\u0000\u0000\u0000\u032f\u0237\u0001\u0000\u0000"+
		"\u0000\u032f\u0239\u0001\u0000\u0000\u0000\u032f\u023b\u0001\u0000\u0000"+
		"\u0000\u032f\u023d\u0001\u0000\u0000\u0000\u032f\u023f\u0001\u0000\u0000"+
		"\u0000\u032f\u0241\u0001\u0000\u0000\u0000\u032f\u0243\u0001\u0000\u0000"+
		"\u0000\u032f\u0245\u0001\u0000\u0000\u0000\u032f\u0247\u0001\u0000\u0000"+
		"\u0000\u032f\u0249\u0001\u0000\u0000\u0000\u032f\u024b\u0001\u0000\u0000"+
		"\u0000\u032f\u024d\u0001\u0000\u0000\u0000\u032f\u024f\u0001\u0000\u0000"+
		"\u0000\u032f\u0251\u0001\u0000\u0000\u0000\u032f\u0253\u0001\u0000\u0000"+
		"\u0000\u032f\u0255\u0001\u0000\u0000\u0000\u032f\u0257\u0001\u0000\u0000"+
		"\u0000\u032f\u0259\u0001\u0000\u0000\u0000\u032f\u025b\u0001\u0000\u0000"+
		"\u0000\u032f\u025d\u0001\u0000\u0000\u0000\u032f\u025f\u0001\u0000\u0000"+
		"\u0000\u032f\u0261\u0001\u0000\u0000\u0000\u032f\u0263\u0001\u0000\u0000"+
		"\u0000\u032f\u0265\u0001\u0000\u0000\u0000\u032f\u0267\u0001\u0000\u0000"+
		"\u0000\u032f\u0269\u0001\u0000\u0000\u0000\u032f\u026b\u0001\u0000\u0000"+
		"\u0000\u032f\u026d\u0001\u0000\u0000\u0000\u032f\u0273\u0001\u0000\u0000"+
		"\u0000\u032f\u0279\u0001\u0000\u0000\u0000\u032f\u027f\u0001\u0000\u0000"+
		"\u0000\u032f\u0285\u0001\u0000\u0000\u0000\u032f\u028d\u0001\u0000\u0000"+
		"\u0000\u032f\u0295\u0001\u0000\u0000\u0000\u032f\u029d\u0001\u0000\u0000"+
		"\u0000\u032f\u02a5\u0001\u0000\u0000\u0000\u032f\u02ad\u0001\u0000\u0000"+
		"\u0000\u032f\u02b5\u0001\u0000\u0000\u0000\u032f\u02bd\u0001\u0000\u0000"+
		"\u0000\u032f\u02c5\u0001\u0000\u0000\u0000\u032f\u02cd\u0001\u0000\u0000"+
		"\u0000\u032f\u02d5\u0001\u0000\u0000\u0000\u032f\u02db\u0001\u0000\u0000"+
		"\u0000\u032f\u02e3\u0001\u0000\u0000\u0000\u032f\u02eb\u0001\u0000\u0000"+
		"\u0000\u032f\u02f3\u0001\u0000\u0000\u0000\u032f\u02fb\u0001\u0000\u0000"+
		"\u0000\u032f\u0303\u0001\u0000\u0000\u0000\u032f\u030b\u0001\u0000\u0000"+
		"\u0000\u032f\u0313\u0001\u0000\u0000\u0000\u032f\u031b\u0001\u0000\u0000"+
		"\u0000\u032f\u0323\u0001\u0000\u0000\u0000\u0330\u0333\u0001\u0000\u0000"+
		"\u0000\u0331\u032f\u0001\u0000\u0000\u0000\u0331\u0332\u0001\u0000\u0000"+
		"\u0000\u0332W\u0001\u0000\u0000\u0000\u0333\u0331\u0001\u0000\u0000\u0000"+
		"\u0334\u0337\u0005\u00c4\u0000\u0000\u0335\u0336\u0005\t\u0000\u0000\u0336"+
		"\u0338\u0003@ \u0000\u0337\u0335\u0001\u0000\u0000\u0000\u0337\u0338\u0001"+
		"\u0000\u0000\u0000\u0338Y\u0001\u0000\u0000\u0000\u0339\u033b\u0005\u00b9"+
		"\u0000\u0000\u033a\u033c\u0003B!\u0000\u033b\u033a\u0001\u0000\u0000\u0000"+
		"\u033b\u033c\u0001\u0000\u0000\u0000\u033c\u033d\u0001\u0000\u0000\u0000"+
		"\u033d\u0353\u0005\u0005\u0000\u0000\u033e\u0340\u0005\u00ba\u0000\u0000"+
		"\u033f\u0341\u0003B!\u0000\u0340\u033f\u0001\u0000\u0000\u0000\u0340\u0341"+
		"\u0001\u0000\u0000\u0000\u0341\u0342\u0001\u0000\u0000\u0000\u0342\u0353"+
		"\u0005\u0005\u0000\u0000\u0343\u0345\u0005\u00bb\u0000\u0000\u0344\u0346"+
		"\u0003B!\u0000\u0345\u0344\u0001\u0000\u0000\u0000\u0345\u0346\u0001\u0000"+
		"\u0000\u0000\u0346\u0347\u0001\u0000\u0000\u0000\u0347\u0353\u0005\u0005"+
		"\u0000\u0000\u0348\u034a\u0005\u00bc\u0000\u0000\u0349\u034b\u0003B!\u0000"+
		"\u034a\u0349\u0001\u0000\u0000\u0000\u034a\u034b\u0001\u0000\u0000\u0000"+
		"\u034b\u034c\u0001\u0000\u0000\u0000\u034c\u0353\u0005\u0005\u0000\u0000"+
		"\u034d\u034f\u0005\u00bd\u0000\u0000\u034e\u0350\u0003B!\u0000\u034f\u034e"+
		"\u0001\u0000\u0000\u0000\u034f\u0350\u0001\u0000\u0000\u0000\u0350\u0351"+
		"\u0001\u0000\u0000\u0000\u0351\u0353\u0005\u0005\u0000\u0000\u0352\u0339"+
		"\u0001\u0000\u0000\u0000\u0352\u033e\u0001\u0000\u0000\u0000\u0352\u0343"+
		"\u0001\u0000\u0000\u0000\u0352\u0348\u0001\u0000\u0000\u0000\u0352\u034d"+
		"\u0001\u0000\u0000\u0000\u0353[\u0001\u0000\u0000\u0000\u0354\u0355\u0005"+
		"\u00c2\u0000\u0000\u0355]\u0001\u0000\u0000\u0000;cekquy\u0080\u0084\u0089"+
		"\u0096\u0099\u009f\u00ab\u00b9\u00c5\u00d1\u00d9\u00e4\u00e9\u00fa\u00fd"+
		"\u0101\u0104\u0110\u0116\u011e\u0124\u0126\u012d\u0131\u013f\u0148\u014e"+
		"\u017f\u0186\u018f\u019f\u01a6\u01a9\u01ae\u01b6\u01b8\u01d7\u01eb\u01ed"+
		"\u01f5\u0200\u0202\u020a\u0211\u032f\u0331\u0337\u033b\u0340\u0345\u034a"+
		"\u034f\u0352";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}