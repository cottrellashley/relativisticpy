r"""The parser: what every piece of notation means, structurally.

Each test states a reading as an S-expression (see ``to_sexpr``). Scripts
print in index form, ``(g _[mu nu])``, when they can be read as indices and in
power form, ``(^ x 2)``, when the exponent is a number.
"""

from textwrap import dedent

import pytest

from relativisticpy.errors import ParseError
from relativisticpy.language import Source, parse, parse_expression, to_sexpr
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels


def sexpr(text: str) -> str:
    return to_sexpr(parse_expression(text))


def program(text: str) -> list[str]:
    return [to_sexpr(statement) for statement in parse(text).statements]


def error(text: str) -> ParseError:
    with pytest.raises(ParseError) as caught:
        parse(text)
    return caught.value


class TestArithmetic:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1 + 2 * 3", "(+ 1 (* 2 3))"),
            ("(1 + 2) * 3", "(* (+ 1 2) 3)"),
            ("1 - 2 - 3", "(- (- 1 2) 3)"),
            ("8 / 4 / 2", "(/ (/ 8 4) 2)"),
            ("2 ** 3 ** 2", "(** 2 (** 3 2))"),
            ("-x ** 2", "(- (** x 2))"),
            ("-x^2", "(- (^ x 2))"),
            ("2 ** -1", "(** 2 (- 1))"),
            ("--x", "(- (- x))"),
            ("+x", "(+ x)"),
        ],
    )
    def test_precedence_and_associativity(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2x", "(* 2 x)"),
            ("2 x y", "(* (* 2 x) y)"),
            ("2(x + 1)", "(* 2 (+ x 1))"),
            ("(a + b)(c + d)", "(* (+ a b) (+ c d))"),
            ("x^2 y^2", "(* (^ x 2) (^ y 2))"),
            ("2|x|", "(* 2 (abs x))"),
            ("a/2b", "(* (/ a 2) b)"),
            ("x [1, 2]", "(* x (list 1 2))"),
        ],
    )
    def test_juxtaposition_is_multiplication(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize("text", ["10 000", r"2\,3", r"10\,00", "1.5.2", "[1 2 3]", "x^2 3"])
    def test_numbers_side_by_side_are_not_a_product(self, text):
        assert "two numbers side by side" in str(error(text))

    @pytest.mark.parametrize(
        ("text", "expected"),
        [(r"10\,000 x", "(* 10000 x)"), (r"299\,792\,458", "299792458"), (r"2\, x", "(* 2 x)")],
    )
    def test_digits_grouped_in_threes_are_one_number(self, text, expected):
        assert sexpr(text) == expected

    def test_juxtaposition_is_marked_implicit(self):
        node = parse_expression("2x")
        assert isinstance(node, ast.Binary)
        assert node.implicit
        assert not parse_expression("2*x").implicit  # type: ignore[attr-defined]

    def test_calls_need_the_parenthesis_against_the_name(self):
        assert sexpr("f(x, y)") == "(call f x y)"
        assert sexpr("f()") == "(call f)"
        assert sexpr("f (x)") == "(* f x)"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # a name numbered by a subscript is a name of its own, called as x_0(t) is
            (r"\psi_1(x)", "(call psi_1 x)"),
            (r"R_{21}(r)", "(call R_21 r)"),
            (r"\psi_{100}(r, \theta)", "(call psi_100 r theta)"),
            (r"\phi_0'(r)", "(call phi_0' r)"),
            ("x_0(t)", "(call x_0 t)"),
            # and as SymPy typesets it, maybe raised to a power
            (r"\psi_{1}{\left(x \right)}", "(call psi_1 x)"),
            (r"\psi_{1}^{2}{\left(x \right)}", "(^ (call psi_1 x) 2)"),
            # a space, an index or a label after the name keeps it a product
            (r"\psi_1 (x)", "(* (psi _[1]) x)"),
            (r"A_\mu(x)", "(* (A _[mu]) x)"),
            (r"h_+(t)", "(* (h _+) t)"),
        ],
    )
    def test_a_numbered_name_is_called_as_a_name_is(self, text, expected):
        assert sexpr(text) == expected

    def test_nested_calls(self):
        assert sexpr("f(g(x))") == "(call f (call g x))"

    def test_primes_are_derivatives_of_functions(self):
        assert sexpr("f'(x)") == "(call f' x)"
        assert sexpr("f'''(x)") == "(call f''' x)"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # a number as subscript makes a name of its own, primed whole, as phi_0' is
            (r"\phi_0' - \phi_0", "(- phi_0' (phi _[0]))"),
            ("y_{1}''", "y_1''"),
            (r"\phi_0'(r)", "(call phi_0' r)"),
            # after a braced script the prime marks the symbol, as TeX sets it
            (r"R_{\mu\nu}'", "(R' _[mu nu])"),
            # after an unbraced label, the label
            (r"x^\mu'", "(x ^[mu'])"),
        ],
    )
    def test_primes_after_scripts(self, text, expected):
        assert sexpr(text) == expected

    def test_a_prime_that_has_no_symbol_to_mark_is_an_error(self):
        assert "a prime after these scripts is ambiguous" in str(error(r"\dot{x}_0'"))

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # (x^3)' was x'^3, and (e^{-x^2})' e'^{-x^2}: the prime went inside the brackets
            ("(x^3)'", "(' (^ x 3))"),
            ("(e^{-x^2})'", "(' (^ e (- (^ x 2))))"),
            (r"\left(\sin x\right)''", "('' (call sin x))"),
            ("(x^2 + 1)'(3)", "(call (' (+ (^ x 2) 1)) 3)"),
            ("{x^3}'", "(^ x' 3)"),  # braces do not show, so the prime is on x, as TeX sets it
        ],
    )
    def test_primes_after_brackets_are_of_the_brackets(self, text, expected):
        assert sexpr(text) == expected

    def test_factorial(self):
        assert sexpr("n!") == "(! n)"
        assert sexpr("(n - k)!") == "(! (- n k))"
        assert sexpr("n! / k!") == "(/ (! n) (! k))"

    def test_double_factorial(self):
        assert sexpr("n!!") == "(call factorial2 n)"  # n(n-2)(n-4)..., not (n!)!
        assert sexpr("(n!)!") == "(! (! n))"
        assert sexpr("n! !") == "(! (! n))"

    def test_absolute_value(self):
        assert sexpr("|x - 3|") == "(abs (- x 3))"
        assert sexpr("|-x^2|") == "(abs (- (^ x 2)))"
        assert sexpr("|x| + |y|") == "(+ (abs x) (abs y))"
        assert sexpr("||x| - 1|") == "(abs (- (abs x) 1))"
        assert sexpr("|f(|x|)|") == "(abs (call f (abs x)))"

    def test_numbers_are_kept_as_written(self):
        node = parse_expression("6.674e-11")
        assert node == ast.Number("6.674e-11", span=node.span)

    def test_strings(self):
        assert sexpr("symbols('x y')") == "(call symbols 'x y')"

    def test_line_breaks_inside_brackets_are_ignored(self):
        assert sexpr("f(\n  x,\n  y\n)") == "(call f x y)"

    def test_a_trailing_operator_continues_the_line(self):
        assert program("y := x +\n  1") == ["(:= y (+ x 1))"]

    def test_an_indented_line_starting_with_an_operator_continues_the_statement(self):
        assert program("y := a\n  + b\n  - c") == ["(:= y (- (+ a b) c))"]
        assert program("ds^2 = -dt^2\n       + dx^2") == ["(= (^ ds 2) (+ (- (^ dt 2)) (^ dx 2)))"]
        assert program("  x\n      = y") == ["(= x y)"]

    def test_an_unindented_line_starting_with_a_minus_is_a_new_statement(self):
        assert program("y := a\n-b") == ["(:= y a)", "(- b)"]

    def test_a_line_starting_with_an_operator_that_starts_no_statement_continues(self):
        # + b on a line of its own is never meant as a statement: it goes on with y
        assert program("  y := a\n  + b") == ["(:= y (+ a b))"]
        assert program("x := 1 + 2\n+ 3\nx") == ["(:= x (+ (+ 1 2) 3))", "x"]
        assert program("E = m\n= n") == ["(chain E = m = n)"]
        assert program("y := a\n\n+ b") != ["(:= y (+ a b))"]  # not after a blank line

    def test_continuation_inside_a_block(self):
        source = "f : (x) -> {\n    y := x\n        + 1\n    y\n}"
        assert program(source) == ["(def f (x) (block (:= y (+ x 1)) y))"]


class TestScripts:
    def test_a_numeric_superscript_is_a_power(self):
        assert sexpr("x^2") == "(^ x 2)"
        assert sexpr("x^{-1}") == "(^ x (- 1))"
        assert sexpr("x^-1") == "(^ x (- 1))"
        assert sexpr("e^{x^2}") == "(^ e (^ x 2))"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # a sign alone is the script, as TeX sets it: h_+ is a polarisation, not h_{+x}
            (r"h_+ \cos[\omega (t - z)]", "(* (h _+) (call cos (* omega (- t z))))"),
            ("h_+ x", "(* (h _+) x)"),
            (r"h_+\cos\omega t", "(* (h _+) (call cos (* omega t)))"),
            (r"C_+ e^{ik_\sigma x^\sigma}", "(* (C _+) (^ e (* (ik _[sigma]) (x ^[sigma]))))"),
            ("r_+r_-", "(* (r _+) (r _-))"),
            ("r_+ + r_-", "(+ (r _+) (r _-))"),
            ("a_-n", "(* (a _-) n)"),
            ("x^+ x^-", "(* (^ x +) (^ x -))"),
            (r"e^+e^- \to \mu^+\mu^-", "(→ (* (^ e +) (^ e -)) (* (^ mu +) (^ mu -)))"),
            # a sign against what it signs is a sign, as plain text writes a power
            ("x^-1", "(^ x (- 1))"),
            ("x^- 1", "(^ x (- 1))"),
            ("10^-3", "(^ 10 (- 3))"),
            ("a_-1", "(a _(- 1))"),
            ("e^-x", "(^ e (- x))"),
            (r"e^-\alpha t", "(* (^ e (- alpha)) t)"),
        ],
    )
    def test_a_sign_as_a_script_is_a_label_unless_it_signs_what_follows(self, text, expected):
        assert sexpr(text) == expected

    def test_braced_scripts_keep_both_readings(self):
        script = parse_expression("x^{a}").scripts[0]  # type: ignore[attr-defined]
        assert script.body == ast.Name("a", span=script.body.span)
        assert [item.label for item in script.indices.items] == ["a"]

    def test_a_script_that_is_not_an_index_list_is_only_an_expression(self):
        script = parse_expression("x^{n+1}").scripts[0]  # type: ignore[attr-defined]
        assert script.indices is None
        assert to_sexpr(script.body) == "(+ n 1)"

    def test_a_script_that_is_not_an_expression_is_only_an_index_list(self):
        script = parse_expression("T_{a:0}").scripts[0]  # type: ignore[attr-defined]
        assert script.body is None
        assert to_sexpr(script.indices) == "[a=0]"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"g_{\mu\nu}", "(g _[mu nu])"),
            ("g_{mu nu}", "(g _[mu nu])"),
            ("g_{mu}_{nu}", "(g _[mu] _[nu])"),
            ("R_{ab}", "(R _[a b])"),
            ("R_{munu}", "(R _[mu nu])"),
            ("R^{a}_{bcd}", "(R ^[a] _[b c d])"),
            (r"R^{\rho}{}_{\sigma\mu\nu}", "(R ^[rho] _[sigma mu nu])"),
            (r"\Gamma^\mu_{\nu\rho}", "(Gamma ^[mu] _[nu rho])"),
            (r"\partial_\mu", "(∂ _[mu])"),
            ("x_{0}", "(x _[0])"),
            ("g_{01}", "(g _[0 1])"),
            ("T_{a1}", "(T _[a1])"),
        ],
    )
    def test_index_lists(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Ric_{mu=0 nu=0}", "(Ric _[mu=0 nu=0])"),
            ("C^{a:1}_{b:0 c:0}", "(C ^[a=1] _[b=0 c=0])"),
            ("R^{a=0}_{b c n=3}", "(R ^[a=0] _[b c n=3])"),
            ("C^{a=t}_{b=r c=r}", "(C ^[a=t] _[b=r c=r])"),
            ("g_{0 0}", "(g _[0 0])"),
        ],
    )
    def test_index_values(self, text, expected):
        assert sexpr(text) == expected

    def test_symmetrisation(self):
        assert sexpr("T_{(ab)}") == "(T _[(a b)])"
        assert sexpr(r"F_{[\mu\nu]}") == "(F _[<mu nu>])"
        assert sexpr("T_{a(bc)}") == "(T _[a (b c)])"
        items = parse_expression("T_{(ab)c}").scripts[0].indices.items  # type: ignore[attr-defined]
        assert [(i.label, i.opens, i.closes) for i in items] == [
            ("a", "(", None),
            ("b", None, ")"),
            ("c", None, None),
        ]

    def test_brackets_may_span_tensors(self):
        assert sexpr(r"\nabla_{[\lambda} R_{\rho\sigma]\mu\nu}") == (
            "(* (∇ _[<lambda]) (R _[rho sigma> mu nu]))"
        )

    def test_brackets_include_derivative_indices(self):
        assert sexpr("F_{[ab,c]}") == "(F _[<a b , c>])"

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("T_{(a)}", "only two or more indices can be (anti)symmetrised"),
            ("T_{((ab))}", "(anti)symmetrisation brackets cannot be nested"),
            ("T_{(a(bc))}", "(anti)symmetrisation brackets cannot be nested"),
            ("T_{()}", "brackets must enclose indices"),
        ],
    )
    def test_why_a_script_is_not_an_index_list_is_recorded(self, text, message):
        # Each of these is still a valid expression, so the parse succeeds; the
        # compiler reports the index error if the script must be indices.
        script = parse_expression(text).scripts[0]  # type: ignore[attr-defined]
        assert script.indices is None
        assert message in str(script.index_error)

    def test_a_subscript_that_is_neither_reports_the_index_error(self):
        assert "mismatched brackets" in str(error("T_{(ab]}"))

    def test_comma_and_semicolon_mark_derivative_indices(self):
        assert sexpr(r"T_{\mu\nu,\rho}") == "(T _[mu nu , rho])"
        assert sexpr("V^{a}{}_{;b}") == "(V ^[a] _[; b])"
        items = parse_expression(r"g_{ab,c}").scripts[0].indices.items  # type: ignore[attr-defined]
        assert [(i.label, i.derivative) for i in items] == [("a", None), ("b", None), ("c", ",")]

    def test_empty_groups_space_the_indices(self):
        assert sexpr("R^{a}{}_{b}{}^{c}") == "(R ^[a] _[b] ^[c])"

    def test_grouping_starts_new_scripts(self):
        assert sexpr("(x^2)^3") == "(^ (^ x 2) 3)"
        assert sexpr("{x^2}^3") == "(^ (^ x 2) 3)"
        assert sexpr("x^2^3") == "(x ^2 ^3)"

    def test_script_after_a_call(self):
        assert sexpr("f(x)^2") == "(^ (call f x) 2)"

    def test_index_items_have_their_own_spans(self):
        items = parse_expression("R_{abc}").scripts[0].indices.items  # type: ignore[attr-defined]
        assert [item.span.text for item in items] == ["a", "b", "c"]


class TestIndexLabels:
    @pytest.mark.parametrize(
        ("name", "labels"),
        [
            ("a", ["a"]),
            ("ab", ["a", "b"]),
            ("abc", ["a", "b", "c"]),
            ("mu", ["mu"]),
            ("munu", ["mu", "nu"]),
            ("theta", ["theta"]),
            ("rtheta", ["r", "theta"]),
            ("alphabeta", ["alpha", "beta"]),
            ("a1", ["a1"]),
            ("x_0", ["x_0"]),
        ],
    )
    def test_split(self, name, labels):
        assert split_index_labels(name) == labels


class TestRelations:
    def test_equation(self):
        assert sexpr("y(x) = 10") == "(= (call y x) 10)"
        assert sexpr(r"G_{\mu\nu} = 8 \pi T_{\mu\nu}") == (
            "(= (G _[mu nu]) (* (* 8 pi) (T _[mu nu])))"
        )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("a = b = c", "(chain a = b = c)"),
            (r"a \approx b = c", "(chain a ≈ b = c)"),
            (
                r"x^\mu \rightarrow x^{\mu'} = x^\mu + a^\mu",
                "(chain (x ^[mu]) → (x ^[mu']) = (+ (x ^[mu]) (a ^[mu])))",
            ),
            # a comparison among equalities is links of the chain, never one operand: a
            # = b \geq c was a = (b >= c), an expression equal to a truth value
            (r"a = b \geq c", "(chain a = b >= c)"),
            (r"a \leq b = c", "(chain a <= b = c)"),
            ("a = 0 < b < c", "(chain a = 0 < b < c)"),
            (r"y = (x \geq 0)", "(= y (>= x 0))"),  # in brackets, one operand
        ],
    )
    def test_relations_chain(self, text, expected):
        assert to_sexpr(parse_expression(text)) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\alpha \sim l_P^2", "(~ alpha (^ l_P 2))"),
            (r"\rho \propto a^{-3}", "(∝ rho (^ a (- 3)))"),
            (r"x \in M", "(∈ x M)"),
            (r"\phi : M \rightarrow N", "(chain phi : M → N)"),
        ],
    )
    def test_relations_that_are_not_computed(self, text, expected):
        assert to_sexpr(parse_expression(text)) == expected

    def test_a_relation_keeps_how_it_was_written(self):
        # so that a message can say \propto, as the user wrote it, rather than ∝
        relation = parse_expression(r"\rho \propto a^{-3}")
        assert isinstance(relation, ast.Relation)
        assert (relation.op, relation.written) == ("∝", r"\propto")
        chain = parse_expression(r"a \approx b ≈ c")
        assert isinstance(chain, ast.Chain)
        assert [link.written for link in chain.links()] == [r"\approx", "≈"]  # type: ignore[union-attr]
        identity = parse(r"\nabla \cdot E \equiv \rho").statements[0].expr  # type: ignore[attr-defined]
        assert identity.written == r"\equiv"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("x > 0", "(> x 0)"),
            ("x <= y + 1", "(<= x (+ y 1))"),
            (r"x \geq 0", "(>= x 0)"),
            ("x != 0", "(!= x 0)"),
            ("x == 0", "(== x 0)"),
            ("0 < x <= 1", "(chain 0 < x <= 1)"),
        ],
    )
    def test_comparisons(self, text, expected):
        assert sexpr(text) == expected

    def test_logic(self):
        assert sexpr("x > 0 and y > 0 or z") == "(or (and (> x 0) (> y 0)) z)"
        assert sexpr("not x > 0") == "(not (> x 0))"

    def test_substitution_binds_loosest(self):
        assert sexpr("x^2 + 1 |_{x = 2}") == "(at (+ (^ x 2) 1) (x 2))"
        assert sexpr("y(x) = 10 x |_{x = 10}") == "(at (= (call y x) (* 10 x)) (x 10))"

    def test_several_substitutions(self):
        assert sexpr("f(x, y) |_{x=1, y=2}") == "(at (call f x y) (x 1) (y 2))"

    def test_latex_evaluation_bar(self):
        assert sexpr(r"\left. x^2 \right|_{x=3}") == "(eval (^ x 2) (x 3))"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # \left. and \right| enclose what the bar evaluates, and nothing else
            (r"x + \left. x^2 \right|_{x=2}", "(+ x (eval (^ x 2) (x 2)))"),
            (
                r"\lim_{d \to 0} \left. q d \right|_{q = p/d}",
                "(lim d 0 (eval (* q d) (q (/ p d))))",
            ),
            (
                r"\left( \left. v \right|_{x=1} - \left. v \right|_{x=0} \right)",
                "(- (eval v (x 1)) (eval v (x 0)))",
            ),
            (r"\left( a + b \right|_{a=1}", "(eval (+ a b) (a 1))"),
            # several substitutions, made at once, of what they enclose
            (r"2 \left. x^2 + y \right|_{x=3, y=1}", "(* 2 (at (+ (^ x 2) y) (x 3) (y 1)))"),
            # a bar without \left. evaluates all that precedes it, as before
            ("x + x^2 |_{x=2}", "(at (+ x (^ x 2)) (x 2))"),
            # and the closing bar of \left| x \right| is an absolute value's
            (r"\left| x \right|_2", "((abs x) _[2])"),
        ],
    )
    def test_an_evaluation_bar_closes_what_left_opens(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\left. F \right|_a^b", "(eval F (? a b))"),
            (r"\left. F \right|^{b}_{a}", "(eval F (? a b))"),
            (r"\left. F \right|_{x=a}^{x=b}", "(eval F (x a b))"),
            (r"\left[ x e^x \right]_0^1", "(eval (* x (e ^[x])) (? 0 1))"),
            (r"[x^2]_{x=0}^{x=1}", "(eval (^ x 2) (x 0 1))"),
            (r"\big[ x^2 \big]_0^1", "(eval (^ x 2) (? 0 1))"),
            (r"[x^2]_{x=0}", "(eval (^ x 2) (x 0))"),
            # a point alone, after \right|, for the reader to find its variable
            (r"\left. f'(x) \right|_{x_0}", "(eval (call f' x) (? x_0))"),
            # a bar alone with limits evaluates the sum before it, but not an equation
            (r"\frac{x^3}{3} \Big|_0^1", "(eval (/ (^ x 3) 3) (? 0 1))"),
            (r"x + \frac{x^3}{3} \Big|_0^1", "(eval (+ x (/ (^ x 3) 3)) (? 0 1))"),
            (r"a = x^2 \Big|_0^1", "(= a (eval (^ x 2) (? 0 1)))"),
        ],
    )
    def test_an_evaluation_between_limits(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # a concentration, as chemistry writes it, and a bracket with a label
            (r"[\mathrm{A}]_0^2", "((list A) _[0] ^2)"),
            (r"\left[ \frac{q}{R} \right]_{\text{ret}}", "((/ q R) _[ret])"),
            (r"\left[ x \right]^2", "(^ x 2)"),
            ("[1, 2]_0^1", "((list 1 2) _[0] ^1)"),
        ],
    )
    def test_brackets_that_evaluate_nothing(self, text, expected):
        assert sexpr(text) == expected

    def test_the_limits_of_an_evaluation_are_of_one_variable(self):
        assert "of one variable" in str(error(r"\left. F \right|_{x=0}^{y=1}"))

    def test_substitution_target_can_be_an_expression(self):
        assert sexpr("y |_{diff(f(x), x) = 10}") == "(at y ((call diff (call f x) x) 10))"

    def test_substitution_needs_an_equation(self):
        assert "expected a substitution such as x = 2" in str(error("x |_{2}"))


class TestCollections:
    def test_lists_and_tuples(self):
        assert sexpr("[1, 2, [3, 4]]") == "(list 1 2 (list 3 4))"
        assert sexpr("[]") == "(list)"
        assert sexpr("[1, 2,]") == "(list 1 2)"
        assert sexpr("(x, 0, 1)") == "(tuple x 0 1)"
        assert sexpr("(x,)") == "(tuple x)"
        assert sexpr("()") == "(tuple)"

    def test_multiline_list(self):
        assert sexpr("[\n [1, 0],\n [0, 1]\n]") == "(list (list 1 0) (list 0 1))"

    def test_braces_group(self):
        assert sexpr("{x + 1}") == "(+ x 1)"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # the Poisson bracket with respect to a canonical pair, as its derivatives
            (r"\{q, p\}_{Q, P}", "(- (* (∂/∂Q q) (∂/∂P p)) (* (∂/∂P q) (∂/∂Q p)))"),
            (
                r"\left\{ f, g \right\}_{x, p_x}",
                "(- (* (∂/∂x f) (∂/∂p_x g)) (* (∂/∂p_x f) (∂/∂x g)))",
            ),
            # without a pair, a pair of braces is a pair, as before, and not a bracket
            (r"\{L_x, L_y\}", "(tuple L_x L_y)"),
            (r"\{f, g\}_{PB}", "((tuple f g) _[P B])"),
            ("(q, p)_{Q, P}", "((tuple q p) _[Q , P])"),
        ],
    )
    def test_a_poisson_bracket_names_its_canonical_pair(self, text, expected):
        assert sexpr(text) == expected

    def test_braces_with_commas_are_piecewise(self):
        assert sexpr("{ x**2, x > 0; -x, x <= 0 }") == (
            "(piecewise ((** x 2) if (> x 0)) ((- x) if (<= x 0)))"
        )
        assert sexpr("{ 1, x > 0; 0, otherwise }") == (
            "(piecewise (1 if (> x 0)) (0 if otherwise))"
        )

    @pytest.mark.parametrize("environment", ["matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix"])
    def test_matrix_environments(self, environment):
        source = rf"\begin{{{environment}}} 1 & 2 \\ 3 & 4 \end{{{environment}}}"
        assert sexpr(source) == f"({environment} (1 2) (3 4))"

    def test_matrix_layout_is_free(self):
        source = r"""
            \begin{pmatrix}
                -(1 - \frac{2M}{r}) & 0 \\
                0 & \frac{1}{1 - \frac{2M}{r}} \\
            \end{pmatrix}
        """
        assert sexpr(source) == (
            "(pmatrix ((- (- 1 (/ (* 2 M) r))) 0) (0 (/ 1 (- 1 (/ (* 2 M) r)))))"
        )

    def test_single_row_matrix(self):
        assert sexpr(r"\begin{bmatrix} t & r & \theta & \phi \end{bmatrix}") == (
            "(bmatrix (t r theta phi))"
        )

    def test_array_environment(self):
        assert sexpr(r"\begin{array}{cc} 1 & 2 \end{array}") == "(array (1 2))"

    def test_ragged_matrix_is_an_error(self):
        message = str(error(r"\begin{matrix} 1 & 2 \\ 3 \end{matrix}"))
        assert "row 2 of the matrix has 1 entries but row 1 has 2" in message

    def test_cases(self):
        source = r"\begin{cases} x^2 & \text{if } x > 0 \\ -x & \text{otherwise} \end{cases}"
        assert sexpr(source) == "(piecewise ((^ x 2) if (> x 0)) ((- x) if otherwise))"

    def test_cases_with_comma_before_the_condition(self):
        source = r"\begin{cases} 1, & x \geq 0 \\ 0, & x < 0 \end{cases}"
        assert sexpr(source) == "(piecewise (1 if (>= x 0)) (0 if (< x 0)))"

    @pytest.mark.parametrize(
        "source",
        [
            r"\begin{cases} x, & x > 0, \\ 0, & \text{otherwise.} \end{cases}",
            r"\begin{cases} x & \text{if } x > 0; \\ 0 & \text{otherwise}. \end{cases}",
        ],
    )
    def test_cases_end_their_rows_with_the_punctuation_of_a_sentence(self, source):
        assert sexpr(source) == "(piecewise (x if (> x 0)) (0 if otherwise))"


class TestFractionsRootsAndFunctions:
    def test_fractions(self):
        assert sexpr(r"\frac{1}{2}") == "(/ 1 2)"
        assert sexpr(r"\dfrac{a+b}{c}") == "(/ (+ a b) c)"
        assert sexpr(r"\frac{1}{2} g^{ab}") == "(* (/ 1 2) (g ^[a b]))"

    def test_single_character_arguments_follow_latex(self):
        assert sexpr(r"\frac12") == "(/ 1 2)"
        assert sexpr(r"\frac ab") == "(/ a b)"
        assert sexpr(r"\frac12 x") == "(* (/ 1 2) x)"
        assert sexpr(r"\frac\pi2") == "(/ pi 2)"

    def test_roots(self):
        assert sexpr(r"\sqrt{x}") == "(sqrt x)"
        assert sexpr(r"\sqrt x") == "(sqrt x)"
        assert sexpr(r"\sqrt[3]{x}") == "(root 3 x)"
        assert sexpr(r"\sqrt{-g}") == "(sqrt (- g))"
        assert sexpr("√(x + 1)") == "(sqrt (+ x 1))"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\sin x", "(call sin x)"),
            (r"\sin(x)", "(call sin x)"),
            (r"\sin{x}", "(call sin x)"),
            (r"\sin\theta\cos\phi", "(* (call sin theta) (call cos phi))"),
            (r"\sin 2x", "(call sin (* 2 x))"),
            (r"\sin\omega t", "(call sin (* omega t))"),
            (r"\sin\theta \, r", "(* (call sin theta) r)"),
            (r"\sin^2\theta", "(^ (call sin theta) 2)"),
            (r"\sin^{-1} x", "(call asin x)"),
            (r"\cosh^{-1}(x)", "(call acosh x)"),
            (r"\arctan x", "(call atan x)"),
            (r"\log_{2} x", "(call log x 2)"),
            (r"\ln x + 1", "(+ (call ln x) 1)"),
            (r"\exp(-x^2)", "(call exp (- (^ x 2)))"),
            (r"\exp[-x^2]", "(call exp (- (^ x 2)))"),  # brackets hold the argument too
            (r"\sin[x]", "(call sin x)"),
            (r"\max(a, b)", "(call max a b)"),
            (r"\det g", "(call det g)"),
            (r"\Re(z)", "(call re z)"),
            (r"\ln A(r)", "(call ln (call A r))"),  # a call in an unbracketed argument
            (r"\sin f(x) g(x)", "(call sin (* (call f x) (call g x)))"),
        ],
    )
    def test_function_commands(self, text, expected):
        assert sexpr(text) == expected

    def test_only_log_takes_a_subscript(self):
        assert r"a subscript on \det is not supported" in str(error(r"\det_g A"))

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # \max_v is a maximum over v, not a logarithm to the base v, nor a product
            (r"H(p) = \max_v \left(p v - L(v)\right)",
             "(= (call H p) (max (v) (- (* p v) (call L v))))"),
            (r"\min_{x} f", "(min (x) f)"),
            (r"\sup_{0 \le x < 1} x^n", "(sup (x) (x ^[n]) (chain 0 <= x < 1))"),
            (r"\inf_{x > 0} \left(x + \frac{1}{x}\right)", "(inf (x) (+ x (/ 1 x)) (> x 0))"),
            (r"\min_{x \in \mathbb{R}} x^2", "(min (x) (^ x 2) (∈ x ℝ))"),
            (r"\min_{x, y \in \mathbb{R}} x^2 + y^2",
             "(min (x y) (+ (^ x 2) (^ y 2)) (∈ (tuple x y) ℝ))"),
            (r"\max_{x \in \{0, 1\}^3} x_1", "(max (x) x_1 (∈ x (^ (tuple 0 1) 3)))"),
            # where it is, written in each of the ways books write it
            (r"\arg\min_{w} \; w^{a} \Sigma_{ab} w^{b}",
             "(argmin (w) (* (* (w ^[a]) (Sigma _[a b])) (w ^[b])))"),
            (r"\operatorname*{arg\,max}_k |f_k|", "(argmax (k) (abs f_k))"),
            (r"\operatorname{argmax}_k |f_k|", "(argmax (k) (abs f_k))"),
            (r"\mathrm{arg\,min}_w f", "(argmin (w) f)"),
            (r"\underset{x \in S}{\operatorname{argmin}} f(x)", "(argmin (x) (call f x) (∈ x S))"),
            (r"\min_{\substack{x \in \mathbb{R}^2 \\ x_1 + x_2 = 1}} x_1^2 + x_2^2",
             "(min (x) (+ (^ x_1 2) (^ x_2 2)) (∈ x (^ ℝ 2)) (= (+ x_1 x_2) 1))"),
            # in plain text and Unicode
            ("min_{x} (x^2 + 1)", "(min (x) (+ (^ x 2) 1))"),
            ("argmax_{x >= 0} x*exp(-x)", "(argmax (x) (* x (call exp (- x))) (>= x 0))"),
            ("max_{x ∈ ℝ} f", "(max (x) f (∈ x ℝ))"),
        ],
    )  # fmt: skip
    def test_an_optimum_is_over_the_variables_beneath_it(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # it begins the sum, and is of the whole of it
            (r"\max_x f(x) + 1", "(max (x) (+ (call f x) 1))"),
            (r"a = \min_{x} x^2 + 1", "(= a (min (x) (+ (^ x 2) 1)))"),
            # it is a factor, or has a sign, and is of the factor alone
            (r"2 \max_x f(x) + 1", "(+ (* 2 (max (x) (call f x))) 1)"),
            (r"-\max_x f(x) + 1", "(+ (- (max (x) (call f x))) 1)"),
            (r"\max_{0 \leq p \leq 1} \min_{0 \leq q \leq 1} p q",
             "(max (p) (min (q) (* p q) (chain 0 <= q <= 1)) (chain 0 <= p <= 1))"),
            # brackets around what it is of group it
            (r"\max_{c} \left[\ln c - c\right]", "(max (c) (- (call ln c) c))"),
        ],
    )  # fmt: skip
    def test_an_optimum_is_of_what_follows_it_as_far_as_it_begins(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\min_{x_1, x_2} \; x_1 + x_2 \quad \text{subject to} \quad x_1^2 + x_2^2 = 2",
             "(min (x_1 x_2) (+ x_1 x_2) (= (+ (^ x_1 2) (^ x_2 2)) 2))"),
            (r"\max_{x} \; 3x + 5y \quad \text{s.t.} \quad x \leq 4, \quad x, y \geq 0",
             "(max (x) (+ (* 3 x) (* 5 y)) (<= x 4) (>= (tuple x y) 0))"),
            (r"\min_{x} x^2 \mathrm{s.t.} x \geq 1", "(min (x) (^ x 2) (>= x 1))"),
            ("min_{x} x^2 subject to x >= 1", "(min (x) (^ x 2) (>= x 1))"),
        ],
    )  # fmt: skip
    def test_constraints_follow_subject_to(self, text, expected):
        assert sexpr(text) == expected

    def test_constraints_follow_on_the_lines_after_it_as_books_set_them(self):
        assert program("\\min_{x, y} x + y\n\\text{subject to} \\quad x^2 + y^2 = 2\nz") == [
            "(min (x y) (+ x y) (= (+ (^ x 2) (^ y 2)) 2))",
            "z",
        ]
        aligned = dedent(
            r"""
            \begin{aligned}
            \min_{x} \quad & x_1 + x_2 \\
            \text{s.t.} \quad & x_1^2 + x_2^2 \le 2 \\
            & x_1 \geq 0
            \end{aligned}
            """
        )
        assert program(aligned) == [
            "(min (x) (+ x_1 x_2) (<= (+ (^ x_1 2) (^ x_2 2)) 2) (>= x_1 0))"
        ]

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            (r"\arg\max f", r"\arg\max, where a maximum is, needs the variable it is over"),
            (r"\sup f", r"\sup, the supremum, needs the variable it is over"),
            (r"\max_{x + 1} f", r"beneath \max, x + 1 is no variable"),
            (r"\min_{x}", r"\min_{x} needs what it is the minimum of, after it"),
            (r"\min_{x} x^2 \text{ s.t. } y", r"y is no constraint"),
        ],
    )
    def test_an_optimum_without_what_it_needs_says_what_is_missing(self, text, message):
        assert error(text).message.startswith(message)

    def test_the_extremes_of_values_and_the_argument_of_a_number_are_read(self):
        assert sexpr(r"\max(a, b) + \arg z") == "(+ (call max a b) (call arg z))"
        assert sexpr(r"\mathrm{arg}\, z") == "(* arg z)"

    @pytest.mark.parametrize(
        ("text", "read"),
        [
            (r"{}_2F_1(1, 1; 2; z) = -\frac{\ln(1 - z)}{z}",
             "(= (call _{2}F_1 (list 1 1) 2 z) (- (/ (call ln (- 1 z)) z)))"),
            (r"{}_2F_{1}\left(\tfrac{1}{2}, \tfrac{1}{2}; 2; 1\right)",
             "(call _{2}F_1 (list (/ 1 2) (/ 1 2)) 2 1)"),
            # after a space, an empty group starts a prescript, rather than spacing z's indices
            (r"\arcsin z = z \, {}_2F_1\left(\tfrac{1}{2}, \tfrac{1}{2}; \tfrac{3}{2}; z^2\right)",
             "(= (call asin z) (* z (call _{2}F_1 (list (/ 1 2) (/ 1 2)) (/ 3 2) (^ z 2))))"),
            (r"{}_0F_1(; b; z)", "(call _{0}F_1 (list) b z)"),
            (r"{}_pF_q(a; b; z)", "(call _{p}F_q a b z)"),
        ],
    )  # fmt: skip
    def test_the_hypergeometric_function_separates_its_lists_with_semicolons(self, text, read):
        assert sexpr(text) == read

    def test_the_hypergeometric_function_needs_its_brackets(self):
        assert error(r"{}_2F_1 z").message == (
            "{}_2F_1, the hypergeometric function, needs its parameters in brackets"
        )

    def test_other_prescripts_label_a_symbol(self):
        assert sexpr(r"{}^{(3)}R + {}_2F") == "(+ ^{(3)}R _{2}F)"
        assert sexpr(r"z {}^{(3)}R") == "(* z ^{(3)}R)"
        assert sexpr(r"R^{a}{}_{bcd}") == "(R ^[a] _[b c d])"  # written against it, it spaces

    def test_a_function_argument_stops_at_a_differential(self):
        assert sexpr(r"\sin^2\theta d\phi^2") == "(* (* (^ (call sin theta) 2) d) (^ phi 2))"
        assert sexpr(r"\sin^2\theta\, d\phi^2") == "(* (* (^ (call sin theta) 2) d) (^ phi 2))"

    def test_other_commands(self):
        assert sexpr(r"\binom{n}{k}") == "(call binomial n k)"
        assert sexpr(r"\lfloor x/2 \rfloor") == "(call floor (/ x 2))"
        assert sexpr(r"\left\lceil x \right\rceil") == "(call ceiling x)"
        assert sexpr(r"\infty") == "infty"

    def test_operator_symbols(self):
        assert sexpr(r"\partial_\mu \phi") == "(* (∂ _[mu]) phi)"
        assert sexpr(r"\nabla_a \nabla_b T^{c}") == "(* (* (∇ _[a]) (∇ _[b])) (T ^[c]))"
        assert sexpr(r"∂_μ φ") == "(* (∂ _[mu]) phi)"

    def test_unknown_command_suggests_a_known_one(self):
        message = str(error(r"\farc{1}{2}"))
        assert r"unknown LaTeX command \farc" in message
        assert r"did you mean \frac?" in message

    @pytest.mark.parametrize(
        ("typo", "meant"),
        [
            ("mathrn", "mathrm"),
            ("approxx", "approx"),
            ("displaystile", "displaystyle"),
            ("boldsimbol", "boldsymbol"),
            ("oepratorname", "operatorname"),
            ("Rigtharrow", "Rightarrow"),
            ("nabal", "nabla"),
        ],
    )
    def test_every_command_can_be_suggested(self, typo, meant):
        # these are read by the lexer, so a list of the parser's commands missed them
        assert error(f"x + \\{typo} y").hint == f"did you mean \\{meant}?"

    @pytest.mark.parametrize("name", ["let", "mod", "foo", "inf"])
    def test_no_unrelated_suggestion(self, name):
        # \foo is not \lfloor
        assert "did you mean" not in str(error(f"A^\\{name}"))

    @pytest.mark.parametrize(
        ("text", "message", "hint"),
        [
            (r"\newcommand{\R}{R}", r"\newcommand defines a macro, which a program cannot",
             "relativisticpy check reads the macros of a document"),
            (r"\abs{x}", r"\abs, the physics package's absolute value, is not supported",
             r"write |x|, or \left| x \right|"),
            (r"\norm{x}", r"\norm, the physics package's norm, is not supported",
             r"write \| x \|"),
            (r"\dd x", r"\dd, the physics package's differential, is not supported",
             r"write d x, or \mathrm{d} x"),
        ],
    )  # fmt: skip
    def test_real_latex_that_is_not_supported_says_so(self, text, message, hint):
        caught = error(text)
        assert caught.message == message
        assert caught.hint is not None
        assert caught.hint.endswith(hint)

    def test_an_unknown_environment_lists_the_supported_ones(self):
        caught = error(r"\begin{pmatrx} 1 \end{pmatrx}")
        assert caught.message == "unsupported environment pmatrx"
        assert caught.hint == "did you mean pmatrix?"
        assert caught.notes[0].startswith("the supported environments are matrix, pmatrix")
        assert "cases and dcases for cases" in caught.notes[0]


class TestDerivatives:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Leibniz notation with \frac
            (r"\frac{d}{dx} f(x)", "(* (d/dx _) (call f x))"),
            (r"\frac{df}{dx}", "(d/dx f)"),
            (r"\frac{dy}{dx}", "(d/dx y)"),
            (r"\frac{d f(x)}{d x}", "(d/dx (call f x))"),
            (r"\frac{d^2 y}{dx^2}", "(d^2/dx^2 y)"),
            (r"\frac{d^2 y}{d x^2}", "(d^2/dx^2 y)"),
            (r"\frac{d^2{x^2}}{d{\tau}^2}", "(d^2/dtau^2 (^ x 2))"),
            (r"\frac{\mathrm{d} y}{\mathrm{d} t}", "(d/dt y)"),
            (r"\frac{\partial f}{\partial x}", "(∂/∂x f)"),
            (r"\frac{\partial^2 f}{\partial x^2}", "(∂^2/∂x^2 f)"),
            (r"\frac{\partial^2 f}{\partial x \partial y}", "(∂^2/∂x∂y f)"),
            (r"\frac{\partial^2{x^2}}{\partial{\tau}^2}", "(∂^2/∂tau^2 (^ x 2))"),
            (r"\frac{\partial}{\partial r}", "(∂/∂r _)"),
            # d with brackets against it, which alone would be a call of d
            (r"\frac{d(\sin^2\theta)}{d\theta}", "(d/dtheta (^ (call sin theta) 2))"),
            (r"\frac{\mathrm{d}\left(x^2\right)}{\mathrm{d}x}", "(d/dx (^ x 2))"),
            # as SymPy typesets them: a function's arguments in braces, and a derivative
            # of the function after it alone
            (r"a{\left(t \right)}", "(call a t)"),
            (r"a^{2}{\left(t \right)}", "(^ (call a t) 2)"),
            (r"f{\left(t,r \right)}", "(call f t r)"),
            (r"\frac{d}{d t} a{\left(t \right)} b", "(* (d/dt (call a t)) b)"),
            (r"\frac{d^{2}}{d t^{2}} a^{2}{\left(t \right)}", "(d^2/dt^2 (^ (call a t) 2))"),
            (r"\frac{d}{dx} x (x + 1)", "(* (* (d/dx _) x) (+ x 1))"),  # only SymPy's form
            # the physics package
            (r"\pdv{f}{x}", "(∂/∂x f)"),
            (r"\pdv[3]{f}{x}", "(∂^3/∂x^3 f)"),
            (r"\pdv{f}{x}{y}", "(∂^2/∂x∂y f)"),
            (r"\pdv{x}(x^2 y)", "(* (∂/∂x _) (* (^ x 2) y))"),
            (r"\dv{f}{t}", "(d/dt f)"),
            (r"\dv[2]{f}{t}", "(d^2/dt^2 f)"),
            # plain-text differentials
            ("d{f(x)}/d{x}", "(d/dx (call f x))"),
            ("(d{f(x)})/(d{x})", "(d/dx (call f x))"),
            ("d^3{f(x)}/d{x}^3", "(d^3/dx^3 (call f x))"),
            ("d^{2}{f}/(d{x} d{y})", "(d^2/dxdy f)"),
            # Lagrange notation
            ("f'(x)", "(call f' x)"),
        ],
    )
    def test_notations(self, text, expected):
        assert sexpr(text) == expected

    def test_a_fraction_that_is_not_a_derivative(self):
        assert sexpr(r"\frac{d(x + y)}{2}") == "(/ (call d (+ x y)) 2)"
        assert sexpr(r"\frac{d}{2}") == "(/ d 2)"
        assert sexpr(r"\frac{dx}{2}") == "(/ dx 2)"
        assert sexpr(r"\frac{a}{dx}") == "(/ a dx)"

    def test_orders_must_agree(self):
        message = str(error(r"\frac{d^2 f}{dx}"))
        assert "order is 2 in the numerator but 1 in the denominator" in message
        message = str(error("d^2{f}/d{x}"))
        assert "order is 2 in the numerator but 1 in the denominator" in message

    def test_d_and_partial_cannot_mix(self):
        assert "cannot mix d and \\partial" in str(error(r"\frac{d f}{\partial x}"))


class TestSumsLimitsIntegrals:
    def test_sums_and_products(self):
        assert sexpr(r"\sum_{n=0}^{10} n") == "(sum (n 0 10) n)"
        assert sexpr(r"\sum^{N}_{i=1} a_i") == "(sum (i 1 N) a_i)"
        assert sexpr(r"\sum_{n=1}^\infty \frac{1}{n^2}") == "(sum (n 1 infty) (/ 1 (^ n 2)))"
        assert sexpr(r"\prod_{k=1}^{n} k") == "(prod (k 1 n) k)"
        assert sexpr("∑_{i=0}^{9} i") == "(sum (i 0 9) i)"

    def test_the_body_of_a_sum_is_a_term(self):
        assert sexpr(r"\sum_{i=1}^{n} 2 i + 1") == "(+ (sum (i 1 n) (* 2 i)) 1)"

    def test_sum_needs_its_limits(self):
        assert r"\sum needs limits" in str(error(r"\sum x"))

    def test_sum_needs_its_variable(self):
        assert "must name its variable" in str(error(r"\sum_{0}^{1} x"))

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\lim_{x \to 0} \frac{\sin x}{x}", "(lim x 0 (/ (call sin x) x))"),
            (r"\lim_{x -> 0} sin(x)/x", "(lim x 0 (/ (call sin x) x))"),
            (r"\lim_{x \rightarrow \infty} 1/x", "(lim x infty (/ 1 x))"),
            (r"\lim_{x \to 0^+} 1/x", "(lim x 0+ (/ 1 x))"),
            (r"\lim_{x \to 0^{-}} 1/x", "(lim x 0- (/ 1 x))"),
            (r"\lim_{r \to \infty} a r^2 + G M", "(+ (lim r infty (* a (^ r 2))) (* G M))"),
        ],
    )
    def test_limits(self, text, expected):
        assert sexpr(text) == expected

    def test_limit_needs_an_arrow(self):
        assert "expected a limit such as x \\to 0" in str(error(r"\lim_{x} x"))

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\int f(x) \, dx", "(int (call f x) x)"),
            (r"\int f(x) dx", "(int (call f x) x)"),
            (r"\int f(x)\, \mathrm{d}x", "(int (call f x) x)"),
            (r"\int x^2 + 1 \, dx", "(int (+ (^ x 2) 1) x)"),
            (r"\int_0^1 x \, dx", "(int x (x 0 1))"),
            (r"\int_{0}^{\infty} e^{-x} dx", "(int (^ e (- x)) (x 0 infty))"),
            (r"\int_{-1}^{1} x \, dx", "(int x (x (- 1) 1))"),
            (r"\int \sin\theta \, d\theta", "(int (call sin theta) theta)"),
            (r"\int dx \, e^{-x^2}", "(int (^ e (- (^ x 2))) x)"),
            (r"\iint_0^1 f(x, y) \, dx \, dy", "(int (call f x y) (x 0 1) (y 0 1))"),
            (r"\iiint f \, dx \, dy \, dz", "(int f x y z)"),
            (r"\int_0^1 \int_0^2 x y \, dy \, dx", "(int (int (* x y) (y 0 2)) (x 0 1))"),
            ("∫_0^1 f(x) dx", "(int (call f x) (x 0 1))"),
            (r"\int f(x) dx + 1", "(+ (int (call f x) x) 1)"),
            (r"2 \int x dx", "(* 2 (int x x))"),
        ],
    )
    def test_integrals(self, text, expected):
        assert sexpr(text) == expected

    def test_integral_needs_a_differential(self):
        message = str(error(r"\int x^2"))
        assert "expected a differential such as dx to end the integral" in message

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # the integral ends with the term whose numerator holds its differential
            (r"\int_1^\infty \frac{dr}{r^2} - \pi", "(- (int (/ 1 (^ r 2)) (r 1 infty)) pi)"),
            (r"\int \frac{dx}{x} + C", "(+ (int (/ 1 x) x) C)"),
            (r"\int \frac{dr}{r^2} f(r) - g", "(- (int (* (/ 1 (^ r 2)) (call f r)) r) g)"),
            (r"\int \frac{x\,dx}{1 + x^2} = 1", "(= (int (/ x (+ 1 (^ x 2))) x) 1)"),
            # a differential written apart from its variable, or upright
            (r"\int \frac{d\theta}{\sin\theta}", "(int (/ 1 (call sin theta)) theta)"),
            (r"\int_0^1 \frac{\mathrm{d}x}{1 + x}", "(int (/ 1 (+ 1 x)) (x 0 1))"),
            # products of fractions, and volume elements, as physics writes them
            (
                r"\int \frac{d^3 p}{(2\pi)^3} \frac{1}{2E}",
                "(int (* (/ 1 (^ (* 2 pi) 3)) (/ 1 (* 2 E))) (d^3 p))",
            ),
            (
                r"\int \frac{dp^0}{2\pi} f(p^0)",
                "(int (* (/ 1 (* 2 pi)) (call f (^ p 0))) (^ p 0))",
            ),
            # without a differential in a numerator, the integrand runs to its differential
            (r"\int x^2 + \frac{1}{x} \, dx", "(int (+ (^ x 2) (/ 1 x)) x)"),
        ],
    )
    def test_a_differential_in_a_numerator(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # an evaluation bar before the differential is inside the integrand
            (
                r"\int_0^1 \left. f(x, y) \right|_{x=0} dy",
                "(int (eval (call f x y) (x 0)) (y 0 1))",
            ),
            (
                r"\int_0^{\pi} \left. \sqrt{g} \right|_{r = r_+} d\theta",
                "(int (eval (sqrt g) (r (r _+))) (theta 0 pi))",
            ),
            # and after it, outside
            (r"\int_0^1 x y \, dx \Big|_{y=2}", "(at (int (* x y) (x 0 1)) (y 2))"),
            (r"\left. \int_0^1 x y \, dx \right|_{y=2}", "(eval (int (* x y) (x 0 1)) (y 2))"),
        ],
    )
    def test_an_evaluation_bar_in_an_integral(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # a subscript or a script written against the region: no limit, but a region
            (
                r"\iint_{x^2 + y^2 \le 1} f \, dA",
                "(int f A over (<= (+ (^ x 2) (^ y 2)) 1))",
            ),
            (r"\iiint_V \rho \, dV", "(int rho V over V)"),
            (r"\iint_S \omega", "(int omega over S)"),
        ],
    )
    def test_a_multiple_integral_over_a_region(self, text, expected):
        assert sexpr(text) == expected

    def test_a_differential_of_a_process_is_not_read_as_one_of_a_variable(self):
        # dB_s, of Brownian motion, makes an Ito integral, which ordinary calculus would get
        # wrong: unread, it is refused rather than computed as an integral over B_s
        assert "expected a differential" in str(error(r"\int_0^t B_s \, dB_s"))

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\int_0^L dx", "(int 1 (x 0 L))"),  # a differential alone integrates 1
            (r"\int d^4x", "(int 1 (d^4 x))"),
            (r"\int d^{4}x \sqrt{-g}", "(int (sqrt (- g)) (d^4 x))"),
            (r"\int f(x) {\rm d}x", "(int (call f x) x)"),
            (r"\int {\rm d}x\, f(x)", "(int (call f x) x)"),
            (r"\int dx\, dy\, f(x, y)", "(int (call f x y) x y)"),
            (r"\int f(t') dt'", "(int (call f t') t')"),
            (r"\int_M d\omega", "(int (* d omega) over M)"),  # still a form over a region
            (r"\frac{dx'}{dt}", "(d/dt x')"),
            (r"\frac{dx}{dt'}", "(d/dt' x)"),
            # of a numbered variable, and of a component, not a volume element d^m u
            (r"\int f(x_1) \, dx_1", "(int (call f x_1) x_1)"),
            (r"\frac{dy}{dx_1}", "(d/dx_1 y)"),
            (r"\int A_\mu dx^\mu", "(int (A _[mu]) (x ^[mu]))"),
            (r"\int dp^0 f(p^0)", "(int (call f (^ p 0)) (^ p 0))"),
            # a volume element over primed coordinates, however it is written
            (r"\int d^3x' \, f(x')", "(int (call f x') (d^3 x'))"),
            (r"\int \frac{d^3 x'}{(2\pi)^3} f", "(int (* (/ 1 (^ (* 2 pi) 3)) f) (d^3 x'))"),
        ],
    )
    def test_differentials(self, text, expected):
        assert sexpr(text) == expected

    def test_the_variable_of_a_differential_spans_the_variable(self):
        integral = parse_expression(r"\int f(t') dt'")
        assert integral.bounds[0].variable.span.text == "t'"  # type: ignore[attr-defined]


class TestStatements:
    def test_definitions(self):
        assert program("x := 2") == ["(:= x 2)"]
        assert program(r"x \equiv 2") == ["(:= x 2)"]
        assert program(r"x \coloneqq 2") == ["(:= x 2)"]
        assert program("x ≔ 2") == ["(:= x 2)"]

    def test_tensor_definitions(self):
        assert program("g_{mu nu} := [[1, 0], [0, 1]]") == [
            "(:= (g _[mu nu]) (list (list 1 0) (list 0 1)))"
        ]
        assert program("ds^2 := -dt^2 + dx^2") == ["(:= (^ ds 2) (+ (- (^ dt 2)) (^ dx 2)))"]

    def test_destructuring(self):
        assert program("x, y := symbols('x y')") == ["(:= x y (call symbols 'x y'))"]

    def test_function_definitions(self):
        assert program("f(x) := x^2") == ["(def f (x) (^ x 2))"]
        assert program("h(x, y) := x y") == ["(def h (x y) (* x y))"]
        assert program("f() := 42") == ["(def f () 42)"]
        assert program(r"R_{21}(r) := r e^{-r}") == ["(def R_21 (r) (* r (^ e (- r))))"]

    def test_primes_on_the_left_are_part_of_the_names(self):
        # decision 7 of design/notation.md: f' is a function of its own, t' a parameter
        assert program("f'(x) := 1") == ["(def f' (x) 1)"]
        assert program("r(t') := 2 t'") == ["(def r (t') (* 2 t'))"]

    def test_a_name_with_scripts_and_arguments_is_defined(self):
        # without a command too: the compiler reads R_{nl}(r) as a function's name
        assert program("R_{nl}(r) := r") == ["(:= (* (R _[n l]) r) r)"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # a primed name, as a coordinate of another frame, is a name of its own
            ("x' := 2 t", "(:= x' (* 2 t))"),
            (r"x' := \gamma (x - v t)", "(:= x' (* gamma (- x (* v t))))"),
            (r"\mathbf{E}' := (E_x, E_y)", "(:= E' (tuple E_x E_y))"),
            # so is a name under an accent that is no time derivative
            (r"\hat{p} := 2", "(:= (hat p) 2)"),
            (r"\bar{x} \equiv \frac{a + b}{2}", "(:= (bar x) (/ (+ a b) 2))"),
            # a dot is a time derivative: := defines it or the compiler refuses, never an
            # identity (see tests/reading/test_messages.py)
            (r"\dot{x} := v", "(:= (dot x) v)"),
        ],
    )
    def test_decorated_names_are_defined(self, text, expected):
        assert program(text) == [expected]

    def test_arrow_functions(self):
        source = """
            f : (x, y) -> {
                m := x y
                m^2
            }
        """
        assert program(source) == ["(def f (x y) (block (:= m (* x y)) (^ m 2)))"]

    def test_arrow_function_with_an_expression_body(self):
        assert program("sq : (x) -> x^2") == ["(def sq (x) (^ x 2))"]

    def test_nested_arrow_functions(self):
        source = """
            f : (x) -> {
                g : (y) -> {
                    y^2
                }
                x + g(x)
            }
        """
        assert program(source) == [
            "(def f (x) (block (def g (y) (block (^ y 2))) (+ x (call g x))))"
        ]

    def test_function_body_cannot_be_empty(self):
        assert "a function body cannot be empty" in str(error("f : (x) -> { }"))

    def test_statement_separators(self):
        assert program("a\nb; c \\\\ d") == ["a", "b", "c", "d"]
        assert program(r"a \newline b") == ["a", "b"]
        assert program("\n\n a \n\n\n b \n\n") == ["a", "b"]

    def test_comments(self):
        source = """
            # the metric
            g_{mu nu} := [[1, 0], [0, 1]]  # flat
            % a LaTeX comment
            g_{mu nu}
        """
        assert program(source) == ["(:= (g _[mu nu]) (list (list 1 0) (list 0 1)))", "(g _[mu nu])"]

    def test_alignment_environments_hold_statements(self):
        source = r"""
            \begin{align}
                x &= 1 + 2 \\
                  & + 3 \\
                y &:= x^2
            \end{align}
        """
        # A row that starts with an operator after '&' continues the row before.
        assert program(source) == ["(= x (+ (+ 1 2) 3))", "(:= y (^ x 2))"]

    def test_rows_of_a_derivation_chain(self):
        source = r"""
            \begin{eqnarray}
                a &=& b + c \nonumber \\
                  &=& d \ , \\
                - e &=& f
            \end{eqnarray}
        """
        assert program(source) == ["(chain a = (+ b c) = d)", "(= (- e) f)"]

    def test_an_equation_may_span_lines_inside_an_alignment(self):
        source = r"""
            \begin{equation}
                G_{\mu\nu}
                    = 8 \pi T_{\mu\nu}
            \end{equation}
        """
        assert program(source) == ["(= (G _[mu nu]) (* (* 8 pi) (T _[mu nu])))"]

    def test_math_delimiters_are_ignored(self):
        assert program("$$ E = m c^2 $$") == ["(= E (* m (^ c 2)))"]
        assert program(r"\[ x := 1 \]") == ["(:= x 1)"]

    def test_latex_statements_are_marked(self):
        statements = parse("a := GM\nb := \\frac{GM}{r}\nc := μ\nd := x").statements
        assert [s.latex for s in statements] == [False, True, True, False]

    @pytest.mark.parametrize(
        "text",
        [
            r"\begin{equation} E = mc^2 \end{equation}",
            r"\begin{align} a &= bc \\ d &= ef \end{align}",
            r"$$ (a+b)^2 = a^2 + 2ab + b^2 $$",
            r"\[ E = mc^2 \]",
        ],
    )
    def test_every_statement_of_a_display_is_latex(self, text):
        # even without a command of its own: in a display mc is m times c
        assert all(statement.latex for statement in parse(text).statements)

    def test_a_command_that_leaves_no_token_makes_a_statement_latex(self):
        # \, and {\rm d} leave no token of their own; plain text has no backslash
        sources = [r"a = N(t)\,dt", r"-N^2 {\rm d}t^2", r"\mathrm{d}x", "E = mc^2"]
        assert [parse(s).statements[0].latex for s in sources] == [True, True, True, False]

    def test_a_source_can_be_latex_throughout(self):
        from relativisticpy.language import Source

        statements = parse(Source("E = mc^2\nGM", latex=True)).statements
        assert [s.latex for s in statements] == [True, True]

    def test_column_pairs_of_an_align_are_equations(self):
        source = r"\begin{align} a &= 1 & b &= 2 \\ c &= 3 & d &= 4 \end{align}"
        assert program(source) == ["(= a 1)", "(= b 2)", "(= c 3)", "(= d 4)"]
        assert program(r"\begin{flalign} a &= 1 & b &= 2 \end{flalign}") == ["(= a 1)", "(= b 2)"]

    def test_a_bracket_broken_over_rows_continues_the_equation(self):
        source = r"""
            \begin{align}
                y &= \left( a + b \right. \nonumber \\
                  &\qquad \left. + c \right) \\
                z &= 1
            \end{align}
        """
        assert program(source) == ["(= y (+ (+ a b) c))", "(= z 1)"]

    def test_the_rows_of_multline_are_one_equation(self):
        source = r"\begin{multline} y = a + b \\ + c \\ - d \end{multline}"
        assert program(source) == ["(= y (- (+ (+ a b) c) d))"]

    def test_cannot_define_a_number(self):
        message = str(error("2 := x"))
        assert "cannot define a number" in message
        assert "hint: the left of := must be a name" in message

    def test_function_parameters_must_be_names(self):
        assert "function parameters must be names" in str(error("f(2) := 1"))

    def test_commas_separate_statements(self):
        assert program(r"a = b, \quad c = d") == ["(= a b)", "(= c d)"]

    @pytest.mark.parametrize(
        "text",
        [
            r"a = 1 \quad \text{and} \quad b = 2",
            r"a = 1, \text{ and } b = 2",
            r"a = 1 \text{ or } b = 2",
        ],
    )
    def test_words_between_equations_separate_them(self, text):
        assert program(text) == ["(= a 1)", "(= b 2)"]  # not a = (1 and b) = 2

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"a = b \quad \text{where} \quad c = d", ["(= a b)", "(= c d)"]),
            (r"\text{where} \ c \equiv d", ["(:= c d)"]),  # not where times c
            (r"f(x) = x^2 \quad \text{for} \quad x > 0", ["(= (call f x) (^ x 2))", "(> x 0)"]),
            (r"\sin x \text{ for } x > 0", ["(call sin x)", "(> x 0)"]),
            (r"x = 1 \text{ where }", ["(= x 1)"]),
        ],
    )
    def test_words_of_prose_around_equations_are_not_factors(self, text, expected):
        assert program(text) == expected

    def test_words_inside_a_condition_are_logic(self):
        source = r"\begin{cases} 1 & x > 0 \text{ and } x < 1 \\ 0 & \text{otherwise} \end{cases}"
        assert sexpr(source) == "(piecewise (1 if (and (> x 0) (< x 1))) (0 if otherwise))"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"x \equiv r/M, \quad y \equiv t/M", ["(:= x (/ r M))", "(:= y (/ t M))"]),
            (r"x \equiv r/M, y \equiv t/M", ["(:= x (/ r M))", "(:= y (/ t M))"]),
            (r"x := 1 \qquad y = x^2", ["(:= x 1)", "(= y (^ x 2))"]),
            (r"a = b, \quad c \equiv d", ["(= a b)", "(:= c d)"]),
            (r"x, y := 1 \quad z := 2", ["(:= x y 1)", "(:= z 2)"]),
        ],
    )
    def test_a_definition_shares_its_line_with_other_statements(self, text, expected):
        assert program(text) == expected

    @pytest.mark.parametrize("text", ["x := 1, 2", "x, y := 1, 2"])
    def test_a_definition_is_not_followed_by_a_bare_list(self, text):
        assert "unexpected ','" in str(error(text))

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"\mu, \nu = 0, 1, 2, 3", ["(∈ (tuple mu nu) (tuple 0 1 2 3))"]),
            (r"\mu = 0, 1, 2, 3", ["(∈ mu (tuple 0 1 2 3))"]),
            (r"n = 0, 1, 2, \ldots", ["(∈ n (tuple 0 1 2 ⋯))"]),
            (r"i, j = 1", ["(∈ (tuple i j) (tuple 1))"]),
            (
                r"R_{\mu\nu} = 0, \quad \mu, \nu = 0, 1",
                ["(= (R _[mu nu]) 0)", "(∈ (tuple mu nu) (tuple 0 1))"],
            ),
            # equations listed with commas stay equations
            ("a = b, c = d", ["(= a b)", "(= c d)"]),
            ("x = 1, y", ["(= x 1)", "y"]),
        ],
    )
    def test_names_equal_to_a_list_of_values_give_their_range(self, text, expected):
        # read, but not computed: \mu, \nu = 0, 1, 2, 3 must not define \nu as 0
        assert program(text) == expected

    @pytest.mark.parametrize("text", [r"E = m c^2 \ .", "E = m c^2,", r"E = m c^2 .\label{e}"])
    def test_the_punctuation_of_a_sentence_is_ignored(self, text):
        assert program(text) == ["(= E (* m (^ c 2)))"]


class TestErrors:
    def test_unclosed_parenthesis(self):
        assert str(error("f(x, y")) == dedent("""\
            unclosed '('
             --> <input>:1:2
              |
            1 | f(x, y
              |  ^ this '(' is never closed
            hint: add ')' to close it""")

    def test_unexpected_token(self):
        message = str(error("x := 1 y := 2"))
        assert message.startswith("unexpected ':='")
        assert "hint: put each statement on its own line" in message

    def test_missing_operand(self):
        assert "expected an expression after '+', found end of input" in str(error("x := 1 +"))

    def test_missing_script(self):
        assert "expected a superscript after '^', found end of input" in str(error("x^"))

    def test_bad_index(self):
        message = str(error("T_{a=}"))
        assert "expected an index value (a whole number or a coordinate)" in message

    def test_the_reading_that_got_further_reports_the_error(self):
        assert "expected an expression after '+', found '}'" in str(error("x^{2 +}"))

    def test_error_points_into_multiline_source(self):
        message = str(error("x := 1\ny := (2 +\nz := 3"))
        assert "<input>:2:6" in message  # the bracket that is never closed
        assert "3 | z := 3" in message  # and where the parser was still inside it

    def test_misplaced_alignment_environment(self):
        message = str(error(r"x + \begin{align} y \end{align}"))
        assert "cannot appear inside an expression" in message


class TestPythonHabits:
    """Python written as a program would be reads as something else, so it is an error."""

    @pytest.mark.parametrize(
        ("text", "message", "hint"),
        [
            ("f(x)[1]", "f(x)[1] indexes as Python does, which the language does not",
             "put a space before the bracket"),
            ("[i^2 for i in range(3)]", "a list comprehension is not part of the language",
             r"or a sum, as in \sum_{i=0}^{2} i^2"),
            ("import sympy", "Python's import is not part of the language",
             "there is nothing to import"),
            ("from sympy import diff", "Python's import is not part of the language",
             "there is nothing to import"),
            ("def f(x): return x^2", "Python's def is not part of the language",
             "as in f(x) := x^2"),
            ("series(sin(x), x, 0, n=5)",
             "n=5 is a keyword argument, which the language does not have",
             "as in series(sin(x), x, 0, 5)"),
        ],
    )  # fmt: skip
    def test_python_is_an_error_that_says_what_to_write(self, text, message, hint):
        caught = error(text)
        assert caught.message == message
        assert caught.hint is not None
        assert caught.hint.endswith(hint)

    @pytest.mark.parametrize(
        "text",
        [
            "x [1, 2]",  # a product, with a space
            "subs(x^2, x=2)",  # a substitution: x is in the other argument
            "solve(x=2)",  # the first argument is what is acted on
            "solve(x^2 = 4, x)",
            r"\delta S[\phi] = 0",  # a functional, in LaTeX
            r"e^{-2\beta}[r - 1]",
            r"\begin{cases} 1 & for x > 0 \\ 0 & otherwise \end{cases}",
        ],
    )
    def test_what_only_looks_like_python_is_read(self, text):
        assert parse(text).statements

    def test_brackets_after_a_name_are_left_to_the_reader(self):
        # x[n] is a sequence where x is one (design/notation.md, 31): only the reader knows
        assert to_sexpr(parse("x := [1, 2, 3]\nx[1]").statements[1]) == "(* x (list 1))"


class TestWhereErrorsAre:
    """An error is reported where its cause is, and marks the places that explain it."""

    def test_a_trailing_operator_is_blamed_not_the_next_line(self):
        assert str(error("x := 1\ny := x +\nz := 2")) == dedent("""\
            the expression on line 2 ends with '+'
             --> <input>:2:8
              |
            2 | y := x +
              |        ^ line 2 ends with this '+'
            3 | z := 2
              | - so line 3 was read as its continuation
            note: a line that ends with an operator continues on the next line
            hint: complete the expression on line 2, or remove the '+'""")

    def test_an_unclosed_bracket_is_marked_where_it_opens(self):
        assert str(error("Coordinates := [t, r, \\theta, \\phi\ng_{\\mu\\nu} := g")) == dedent("""\
            unclosed '['
             --> <input>:1:16
              |
            1 | Coordinates := [t, r, \\theta, \\phi
              |                ^ this '[' is never closed
            2 | g_{\\mu\\nu} := g
              |            -- still inside the '[' here
            hint: add ']' to close it, at the end of line 1""")

    def test_the_wrong_closer_marks_the_opener(self):
        lines = str(error("f(x, y]")).splitlines()
        assert lines[0] == "expected ')' to close '(', found ']'"
        assert lines[4:] == ["  |  - opened here", "  |       ^ expected ')' here"]

    @pytest.mark.parametrize(
        ("text", "message", "hint"),
        [
            (r"e^{i\pi", "unclosed '{'", r"where the superscript ends, as in e^{i\pi}"),
            (r"\Gamma^{a}_{bc", "unclosed '{'", r"where the subscript ends, as in \Gamma^{a}_{bc}"),
            ("T^{a b", "unclosed '{'", "where the superscript ends, as in T^{a b}"),
        ],
    )
    def test_an_unclosed_script_says_so_not_that_an_index_is_missing(self, text, message, hint):
        caught = error(text)
        assert caught.message == message
        assert caught.span is not None
        assert caught.span.text == "{"
        assert caught.hint is not None
        assert caught.hint.endswith(hint)

    @pytest.mark.parametrize(
        ("text", "message", "label"),
        [
            ("R_{ab}}", "unmatched '}'", "no '{' is open here"),
            ("x + 1)", "unmatched ')'", "no '(' is open here"),
            ("a & b", "'&' outside an environment", ""),
            ("x.y", "'.' is not an operator", ""),
        ],
    )
    def test_a_stray_symbol_is_named_without_talk_of_statements(self, text, message, label):
        caught = error(text)
        assert caught.message == message
        assert caught.span is not None
        assert caught.span.label == label
        assert "statement" not in str(caught.hint)

    def test_a_list_defined_without_brackets(self):
        assert error("Coordinates := t, r").hint == (
            "to define a list, write it in brackets: Coordinates := [t, r]"
        )

    @pytest.mark.parametrize(
        ("text", "empty"),
        [(r"\frac{1}{}", "{}"), (r"\sqrt[]{x}", "[]"), (r"\lim_{x \to } x", r"{x \to }")],
    )
    def test_an_empty_argument_is_marked_where_it_is_empty(self, text, empty):
        span = error(text).span
        assert span is not None
        assert span.text == empty

    def test_a_missing_argument_is_marked_after_the_last_one(self):
        caught = error(r"\frac{1}")
        assert caught.message == r"\frac is missing its denominator"
        assert caught.span is not None
        assert (caught.span.line, caught.span.column) == (1, 9)
        assert caught.hint == r"write it in braces, as in \frac{1}{2}"

    def test_a_missing_operand_is_marked_after_the_operator_not_on_a_blank_line(self):
        caught = error("x := 3\ny +\n\n")
        assert caught.message == "expected an expression after '+', found end of input"
        assert caught.span is not None
        assert (caught.span.line, caught.span.column) == (2, 4)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            (r"\sin", r"\sin needs an argument"),
            (r"\cos^2", r"\cos^2 needs an argument"),
            ("^2", "'^2' has nothing to attach to"),
            ("y + x_{}", "empty subscript"),
        ],
    )
    def test_messages_name_what_is_missing(self, text, message):
        assert error(text).message == message

    def test_an_empty_script_in_latex_is_nothing_as_tex_sets_it(self):
        # as \p{} makes {\partial}_{} where a paper defines \p#1 as {\partial}_{#1}
        assert sexpr(r"{{\partial}_{} \phi \over {\partial}_{} x}") == "(∂/∂x phi)"
        assert to_sexpr(parse(Source("x_{} + 1", latex=True))) == "(+ x 1)"


def recovered(text: str) -> tuple[list[str], list[tuple[str, int]]]:
    """The statements of a program read past its errors, and each error and its line."""
    program = parse(text, recover=True)
    errors = []
    for caught in program.errors:
        assert caught.span is not None
        errors.append((caught.message, caught.span.line))
    return [to_sexpr(statement) for statement in program.statements], errors


class TestRecovery:
    """parse(source, recover=True) goes on past each statement that cannot be read."""

    def test_every_error_and_every_statement_that_parses(self):
        statements, errors = recovered(
            "a := 1\nT_{ab} u^a\nb := 2\n\\frca{1}{2}\nS_{ab} v^b\na + b"
        )
        assert statements == [
            "(:= a 1)", "(* (T _[a b]) (u ^[a]))", "(:= b 2)", "(* (S _[a b]) (v ^[b]))",
            "(+ a b)",
        ]  # fmt: skip
        assert errors == [(r"unknown LaTeX command \frca", 4)]

    def test_errors_of_the_lexer_too(self):
        statements, errors = recovered("x := 1\ny := 2 @ 3\nz := \frac\nw := 4")
        assert statements == ["(:= x 1)", "(:= w 4)"]
        assert errors == [
            ("unexpected character '@'", 2),
            (r"Python turned \f in \frac into a form feed", 3),
        ]

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("Coordinates := [t, r\ng := 1", "unclosed '['"),  # costs its own line only
            ("x := 1 +\ng := 1", "the expression on line 1 ends with '+'"),
            ("x := 1 )\ng := 1", "unmatched ')'"),
            ("\\begin{equation} x + 1\ng := 1", r"\begin{equation} has no matching \end{equation}"),
            ("M := [[1, 2],\n  [3 @ 4],\n  [5, 6]]\ng := 1", "unexpected character '@'"),
            ("\\begin{align} a &= 1 \\\\ b &= \\frca \\end{align}\ng := 1",
             r"unknown LaTeX command \frca"),
            ("y := a\n  + \\frca\n  + b\ng := 1", r"unknown LaTeX command \frca"),
        ],
    )  # fmt: skip
    def test_one_error_for_one_statement_and_the_next_is_read(self, text, message):
        statements, errors = recovered(text)
        assert [error for error, _ in errors] == [message]
        assert statements[-1] == "(:= g 1)"

    def test_errors_and_statements_in_the_order_they_are_written(self):
        program = parse("x := (1\ny := 2\nz := \\frca", recover=True)
        assert [type(item).__name__ for item in program.in_order()] == [
            "ParseError",
            "Define",
            "ParseError",
        ]

    def test_without_recover_the_first_error_is_raised(self):
        with pytest.raises(ParseError, match="unclosed"):
            parse("x := (1\ny := \\frca")
        assert parse("x := 1").errors == ()


class TestSpans:
    def test_nodes_know_their_source(self):
        node = parse_expression("1 + f(x)")
        assert node.span.text == "1 + f(x)"
        assert node.right.span.text == "f(x)"  # type: ignore[attr-defined]

    def test_spans_do_not_affect_equality(self):
        assert parse_expression("x + 1") == parse_expression("x   +   1")

    def test_statement_spans(self):
        statement = parse("\n  x := 2 y\n").statements[0]
        assert statement.span.text == "x := 2 y"

    @pytest.mark.parametrize(
        "text",
        [
            "(a+b)^2 = a^2 + 2ab + b^2",
            "{x} + 1",
            r"\left( a \right) b",
            "(a + b) c = d",
            "y = (a + b)",
            r"x = {\cos\theta \over \sin\theta}",
        ],
    )
    def test_spans_keep_their_brackets(self, text):
        assert parse(text).statements[0].span.text == text


class TestMoreNotation:
    def test_parse_accepts_a_named_source(self):
        from relativisticpy.language import Source

        assert to_sexpr(parse(Source("x := 1", name="a.relpy"))) == "(:= x 1)"
        assert to_sexpr(parse_expression(Source("x + 1"))) == "(+ x 1)"

    def test_a_whole_program_prints_one_statement_per_line(self):
        assert to_sexpr(parse("x := 1\ny")) == "(:= x 1)\ny"

    def test_scripts_in_parentheses_and_commands(self):
        assert sexpr("x^(n+1)") == "(^ x (+ n 1))"
        assert sexpr(r"e^\frac{1}{2}") == "(^ e (/ 1 2))"
        assert sexpr("x^1.5") == "(^ x 1.5)"

    @pytest.mark.parametrize(("text", "what"), [("x^{}", "superscript"), ("x_{}", "subscript")])
    def test_an_empty_script_is_an_error(self, text, what):
        # x_{} was silently the symbol x_, and x^{} "x is not a tensor"
        assert error(text).message == f"empty {what}"

    def test_a_decimal_is_not_an_index(self):
        script = parse_expression("T_{1.5}").scripts[0]  # type: ignore[attr-defined]
        assert script.indices is None
        assert "an index value must be a whole number" in str(script.index_error)

    def test_trailing_commas_and_semicolons(self):
        assert sexpr("f(x,)") == "(call f x)"
        assert sexpr("{ 1, x > 0; 0, otherwise; }") == "(piecewise (1 if (> x 0)) (0 if otherwise))"

    def test_juxtaposition_with_environments_and_bars(self):
        assert sexpr(r"2 \begin{pmatrix} 1 \end{pmatrix}") == "(* 2 (pmatrix (1)))"
        assert sexpr(r"\sin|x|") == "(call sin (abs x))"

    def test_accents(self):
        assert sexpr(r"\dot{a}") == "(dot a)"
        assert sexpr(r"\ddot x") == "(ddot x)"
        assert sexpr(r"\bar{z} + \hat{n} + \vec{v} + \tilde{\phi}") == (
            "(+ (+ (+ (bar z) (hat n)) (vec v)) (tilde phi))"
        )
        assert sexpr(r"\dot{x}^\mu") == "((dot x) ^[mu])"

    def test_plain_differentials(self):
        assert sexpr("d{x}") == "(d x)"
        assert sexpr("d^2{f}") == "(d^2 f)"
        assert sexpr("d^{2}{f}/d{x}^2") == "(d^2/dx^2 f)"

    def test_a_derivative_of_a_product(self):
        assert sexpr(r"\frac{d f g}{dx}") == "(d/dx (* f g))"

    def test_differential_in_braces_inside_an_integral(self):
        assert sexpr(r"\int f d{x}") == "(int f x)"

    def test_negative_integration_limit_without_braces(self):
        assert sexpr(r"\int_-1^1 x dx") == "(int x (x (- 1) 1))"

    def test_a_denominator_mixing_d_and_partial_is_a_plain_quotient(self):
        assert sexpr(r"\frac{d f}{dx \partial y}") == "(/ (* d f) (* (* dx ∂) y))"

    def test_norm_bars(self):
        assert sexpr(r"\|x\|") == "(abs x)"
        assert sexpr(r"\lVert A \rVert_2") == "((abs A) _[2])"
        assert sexpr("‖x‖_1") == "((abs x) _[1])"


class TestMatrixNotation:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"A^\top", "(A ^[top])"),
            (r"A^\intercal", "(A ^[intercal])"),
            (r"A^\dagger", "(A ^[dagger])"),
            ("Aᵀ", "(A ^[T])"),
            ("A† B", "(* (A ^[dagger]) B)"),
            ("A^† B", "(* (A ^[dagger]) B)"),
        ],
    )
    def test_a_transpose_or_a_dagger_is_a_superscript(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("A^{+} b", "(* (^ A +) b)"),
            ("r_{-}", "(r _-)"),
            ("A^{*}", "(^ A *)"),
            (r"\lambda_{\max}(A)", "(* (lambda _[max]) A)"),
            (r"\lambda_\min(A)", "(* (lambda _[min]) A)"),
        ],
    )
    def test_a_sign_or_an_extreme_alone_in_a_script_is_a_label(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"A \oplus B", "(⊕ A B)"),
            ("A ⊕ B C + D", "(+ (⊕ A (* B C)) D)"),
            (r"\det A \oplus B", "(⊕ (call det A) B)"),
            (r"M_\odot", "(M _[odot])"),
        ],
    )
    def test_a_direct_sum_binds_as_a_sum(self, text, expected):
        assert sexpr(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"(x \circledast h)[0]", "(* (⊛ x h) (list 0))"),
            (r"x[n] \circledast h[n]", "(⊛ (* x (list n)) (* h (list n)))"),
            (r"2 x[n] \circledast h[n] + 1", "(+ (⊛ (* (* 2 x) (list n)) (* h (list n))) 1)"),
            (r"x \circledast_N h", "(⊛_N x h)"),
            (r"x \circledast_{4} h", "(⊛_4 x h)"),
            ("x ⊛ h", "(⊛ x h)"),
            # an asterisk is a product's, for the reader to decide by what is either side
            (r"f(t) \ast g(t)", "(* (call f t) (call g t))"),
            ("f(t) ∗ g(t)", "(* (call f t) (call g t))"),
        ],
    )
    def test_a_circular_convolution_binds_looser_than_a_product(self, text, expected):
        assert sexpr(text) == expected

    def test_the_kernel_and_the_dimension_are_functions(self):
        assert sexpr(r"\dim \ker A") == "(call dim (call null A))"
        assert sexpr(r"\ker(A)") == "(call null A)"


class TestMoreErrors:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("f : (x) x", "expected '->' after the parameters, found name 'x'"),
            ("f : (x) -> { x", "unclosed '{'"),
            ("f : (x) -> {\n", "unclosed '{'"),
            (r"\frac{1}", r"\frac is missing its denominator"),
            (r"\frac{1", r"unclosed '{' in the argument of \frac"),
            (r"\frac{}{2}", r"the numerator of \frac is empty"),
            (r"\frac{1 )}{2}", "unexpected ')' in the numerator"),
            (r"\sum_{n=0}^", r"expected a limit after \sum"),
            (r"\lim x", r"\lim needs a subscript"),
            (r"\pdv[2]{f}{x}{y}", "a mixed derivative cannot also have an order"),
            (r"\pdv[1.5]{f}{x}", "the order of a derivative must be a whole number"),
            ("d^{1.5}{f}/d{x}", "the order of a derivative must be a whole number"),
            (r"\lfloor x", r"expected \rfloor to close \lfloor"),
            (r"\sin^{} x", "empty superscript"),
            (r"\sin + 1", r"\sin needs an argument"),
            (r"\begin{tabular} x \end{tabular}", "unsupported environment tabular"),
            (r"\begin{matrix} 1 , 2 \end{matrix}", r"expected '&', '\\' or \end{matrix}"),
            (r"\begin{matrix} \end{matrix}", "a matrix needs at least one entry"),
            (r"\begin{cases} 1 & x > 0 , 2 \end{cases}", r"expected '\\' or \end{cases}"),
            (r"\begin{cases} \end{cases}", "cases needs at least one case"),
            ("T_{a(}", "brackets must enclose indices"),
            ("5.²", "'.' is not an operator"),  # an error, not a crash in the lexer
        ],
    )
    def test_messages(self, text, message):
        assert message in str(error(text))

    def test_parse_expression_rejects_trailing_input(self):
        with pytest.raises(ParseError, match=r"unexpected '\)'"):
            parse_expression("x )")


class TestDiracNotation:
    """Angle brackets are tokens of their own, and a bar belongs to a bra, a ket or an
    absolute value by what closes it (relativisticpy.language.dirac)."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"|\psi\rangle", "(ket psi)"),
            (r"\langle\phi|", "(bra phi)"),
            (r"\langle\phi|\psi\rangle", "(braket phi psi)"),
            (r"\langle \phi | A | \psi \rangle", "(braket phi A psi)"),
            (r"\langle x \rangle", "(expectation x)"),
            (r"\langle x^2 \rangle", "(expectation (^ x 2))"),
            (r"\langle u, v \rangle", "(expectation (tuple u v))"),
            (r"|+\rangle\langle+|", "(* (ket +) (bra +))"),
            (r"|n - 1\rangle", "(ket (- n 1))"),
            (r"|j, m\rangle", "(ket (tuple j m))"),
            (r"\langle n' | n \rangle", "(braket n' n)"),
            (r"\left| \psi \right\rangle", "(ket psi)"),
            (r"\left\langle \phi \middle| \psi \right\rangle", "(braket phi psi)"),
            ("⟨φ|ψ⟩", "(braket phi psi)"),
            (r"\ket{\psi}", "(ket psi)"),
            (r"\bra{\phi}", "(bra phi)"),
            (r"\braket{\phi|\psi}", "(braket phi psi)"),
            (r"\braket{\phi}{\psi}", "(braket phi psi)"),
            (r"\ketbra{a}{b}", "(* (ket a) (bra b))"),
            (r"\expval{A}", "(expectation A)"),
            (r"\expval{A}{\psi}", "(braket psi A psi)"),
            (r"\mel{a}{A}{b}", "(braket a A b)"),
        ],
    )
    def test_kets_bras_and_brackets(self, text, expected):
        assert sexpr(text) == expected

    def test_a_label_of_groups(self):
        written = (
            r"\langle 1, \tfrac{1}{2}; 1, -\tfrac{1}{2} | 1, \tfrac{1}{2}; \tfrac{3}{2}, 1 \rangle"
        )
        assert sexpr(written) == (
            "(braket (tuple (tuple 1 (/ 1 2)) (tuple 1 (- (/ 1 2)))) "
            "(tuple (tuple 1 (/ 1 2)) (tuple (/ 3 2) 1)))"
        )

    def test_a_bra_alone_is_not_a_matrix_element(self):
        # the + between the outer products is no operator between the bars
        written = r"|+\rangle\langle+| + |-\rangle\langle-|"
        assert sexpr(written) == "(+ (* (ket +) (bra +)) (* (ket -) (bra -)))"

    def test_bars_of_absolute_values_stay_so(self):
        assert sexpr("|x| + |y|") == "(+ (abs x) (abs y))"
        assert sexpr(r"|\langle k | H | 0 \rangle|^2") == "(^ (abs (braket k H 0)) 2)"
        assert sexpr(r"\left. x^2 \right|_{x=1}") == "(eval (^ x 2) (x 1))"

    def test_a_ket_after_a_coefficient_is_a_product(self):
        assert sexpr(r"i|1\rangle - 2|2\rangle") == "(- (* i (ket 1)) (* 2 (ket 2)))"
        assert sexpr(r"a_- |n\rangle") == "(* (a _-) (ket n))"

    def test_kets_brackets_and_angle_brackets_are_defined(self):
        assert program(r"|0\rangle := 1") == ["(:= (ket 0) 1)"]
        assert program(r"\langle x | p \rangle := 1") == ["(:= (braket x p) 1)"]
        assert program(r"\langle x^2 \rangle := 1") == ["(:= (expectation (^ x 2)) 1)"]

    def test_a_star_before_brackets_conjugates_the_call(self):
        assert sexpr(r"\psi^*(x)") == sexpr(r"\psi(x)^*") == "(^ (call psi x) *)"

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            (r"\langle x", r"expected '|' or '\rangle' to close the bracket"),
            (r"|x\rangle\rangle", r"unexpected '\rangle'"),
        ],
    )
    def test_brackets_left_open(self, text, message):
        assert message in str(error(text))


class TestPlainTextAndUnicodeSpellLatex:
    """Plain text and Unicode are spellings of the constructs LaTeX writes with commands:
    each reaches the syntax tree of the LaTeX it spells (design/notation.md, decision 39)."""

    @pytest.mark.parametrize(
        ("spelled", "latex"),
        [
            # sin²x was s i n² x, with i the imaginary unit, and sin 2x was sin(2) x
            ("sin x", r"\sin x"),
            ("sin 2x", r"\sin 2x"),
            ("sin(x) cos(x)", r"\sin(x) \cos(x)"),
            ("sin x cos x", r"\sin x \cos x"),
            ("sin²x + cos²x", r"\sin^2 x + \cos^2 x"),
            ("sin^2 x", r"\sin^2 x"),
            ("sin^(-1) x", r"\sin^{-1} x"),
            ("cos θ", r"\cos\theta"),
            ("x² sin x", r"x^2 \sin x"),
            ("sin x / cos x", r"\sin x / \cos x"),
            ("ln|x|", r"\ln|x|"),
            ("exp(-x^2)", r"\exp(-x^2)"),
            ("log_2 8", r"\log_2 8"),
            ("log_10(x)", r"\log_{10}(x)"),
            ("min_x x^2 - 4x", r"\min_x x^2 - 4x"),
            ("int_0^1 x^2 dx", r"\int_0^1 x^2 \, dx"),
            ("int x^2 dx", r"\int x^2 \, dx"),
            ("∫_0^π sin x dx", r"\int_0^\pi \sin x \, dx"),
            ("lim_{x→0} sin(x)/x", r"\lim_{x \to 0} \sin(x)/x"),
            ("lim_(x->0) sin(x)/x", r"\lim_{x \to 0} \sin(x)/x"),
            ("sum_(n=1)^oo 1/n^2", r"\sum_{n=1}^{oo} 1/n^2"),
            ("sum_{n=1}^oo 1/n^2", r"\sum_{n=1}^{oo} 1/n^2"),
        ],
    )
    def test_the_tree_is_that_of_the_latex(self, spelled, latex):
        assert sexpr(spelled) == sexpr(latex)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("max_iter := 100", ["(:= max_iter 100)"]),
            ("sum := a + b", ["(:= sum (+ a b))"]),
            ("sum(a, b)", ["(call sum a b)"]),
            ("sin + 1", ["(+ sin 1)"]),
        ],
    )
    def test_a_word_that_applies_to_nothing_is_a_name(self, text, expected):
        assert program(text) == expected

    @pytest.mark.parametrize(
        ("spelled", "latex"),
        [
            # d/dx sin(x) was d sin(x)/dx, d(x^3)/dx a function d, and df/dx a name over dx
            ("d/dx sin(x)", r"\frac{d}{dx} \sin(x)"),
            ("d/dx (x^3)", r"\frac{d}{dx} (x^3)"),
            ("d/dx(x^3)", r"\frac{d}{dx}(x^3)"),
            ("(d/dx)(x^3)", r"\frac{d}{dx}(x^3)"),
            ("d^2/dx^2 (x^4)", r"\frac{d^2}{dx^2} (x^4)"),
            ("d²/dx² (x⁴)", r"\frac{d^2}{dx^2} (x^4)"),
            ("d/dθ (sin²θ)", r"\frac{d}{d\theta} (\sin^2\theta)"),
            ("d(x^3)/dx", r"\frac{d(x^3)}{dx}"),
            ("df/dx", r"\frac{df}{dx}"),
            ("df(x)/dx", r"\frac{df(x)}{dx}"),
            ("d^2y/dx^2", r"\frac{d^2y}{dx^2}"),
            ("(dy)/(dx)", r"\frac{dy}{dx}"),
            ("∂f/∂x", r"\frac{\partial f}{\partial x}"),
            ("(del f)/(del x)", r"\frac{\partial f}{\partial x}"),
            ("x dy/dx", r"x \frac{dy}{dx}"),
            ("x*dy/dx", r"x \cdot \frac{dy}{dx}"),
            ("2 d/dx (x^3)", r"2 \frac{d}{dx} (x^3)"),
            ("-d/dx (x^2)", r"-\frac{d}{dx} (x^2)"),
        ],
    )
    def test_a_solidus_between_differentials_is_leibnizs_fraction(self, spelled, latex):
        assert sexpr(spelled) == sexpr(latex)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("dx/2", "(/ dx 2)"),
            ("d/c", "(/ d c)"),
            ("a/dt", "(/ a dt)"),
            ("dr^2/(1 - 2M/r)", "(/ (^ dr 2) (- 1 (/ (* 2 M) r)))"),
        ],
    )
    def test_a_solidus_with_no_differential_on_one_side_divides(self, text, expected):
        assert sexpr(text) == expected

    def test_a_leibniz_solidus_keeps_the_quotient_as_written(self):
        derivative = parse_expression("dy/dx")
        assert isinstance(derivative, ast.Derivative)
        assert derivative.written is not None
        assert to_sexpr(derivative.written.expr) == "(/ dy dx)"
        assert derivative.written.names == ("dy", "dx")

    @pytest.mark.parametrize("text", ["(del f)/(del x)", "(x dy)/dx", "-d/dx (x^2)"])
    def test_leibnizs_fraction_spans_what_is_written(self, text):
        # (del f)/(del x) was shown as del f)/(del x), without its first bracket
        assert parse(text).statements[0].span.text == text

    def test_the_root_sign_takes_the_whole_number_after_it(self):
        # √16 was √1 beside 6, two numbers side by side, as TeX reads \sqrt16
        assert sexpr("√16") == sexpr(r"\sqrt{16}") == "(sqrt 16)"
        assert sexpr("√2x") == "(* (sqrt 2) x)"
