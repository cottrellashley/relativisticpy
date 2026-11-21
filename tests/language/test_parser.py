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
        assert sexpr(r"\left. x^2 \right|_{x=3}") == "(at (^ x 2) (x 3))"

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

    @pytest.mark.parametrize("text", [r"\max_x f(x)", r"\min_{x} f", r"\det_g A"])
    def test_only_log_takes_a_subscript(self, text):
        # \max_x is a maximum over x, not a logarithm to the base x
        command = text.split("_")[0]
        assert f"a subscript on {command} is not supported" in str(error(text))

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


class TestMoreErrors:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("f'(x) := 1", "cannot define this"),
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
            (r"\begin{tabular} x \end{tabular}", "unsupported environment 'tabular'"),
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
