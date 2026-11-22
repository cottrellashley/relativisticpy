r"""General Relativity in the language (ported from the original suite, and extended).

The original tests compared printed strings; these compare mathematically, so
they pin down the physics rather than the form SymPy happens to print.
"""

import pytest
import sympy
from sympy import Array, Function, Matrix, Symbol, cos, diag, diff, sin, symbols

from helpers import same, zero
from relativisticpy import IndexNotationError, Tensor, TensorEquation, spacetimes

t, r, theta, phi, G, M, c = symbols("t r theta phi G M c")
f_ = 1 - 2 * G * M / r
SCHWARZSCHILD = Array(diag(-f_, 1 / f_, r**2, r**2 * sin(theta) ** 2))


class TestMetric:
    def test_definition(self, schwarzschild):
        g = schwarzschild.expr("g_{mu}_{nu}")
        assert isinstance(g, Tensor)
        assert same(g, SCHWARZSCHILD)
        assert str(g.indices) == r"_{\mu\nu}"
        assert g.coordinates == (t, r, theta, phi)

    def test_inverse(self, schwarzschild):
        g = schwarzschild.expr("g^{a}^{b}")
        assert same(g, Array(Matrix(SCHWARZSCHILD.tolist()).inv()))
        assert str(g.indices) == "^{ab}"

    def test_metric_times_its_inverse_is_the_identity(self, schwarzschild):
        delta = schwarzschild.expr("g_{a}_{b}*g^{b}^{c}")
        assert delta.components == Array(sympy.eye(4))
        assert str(delta.indices) == "_{a}^{c}"

    def test_full_contraction_is_the_dimension(self, schwarzschild):
        assert schwarzschild.expr("g_{a}_{b}*g^{a}^{b}") == 4

    def test_bare_g_is_the_determinant(self, schwarzschild):
        assert same(schwarzschild.expr("g"), -(r**4) * sin(theta) ** 2)
        assert same(schwarzschild.expr(r"\sqrt{-g}"), sympy.sqrt(r**4 * sin(theta) ** 2))

    def test_components(self, schwarzschild):
        results = schwarzschild.exe(
            "g_{a:0}_{b:0}\ng_{a:1}_{b:1}\ng_{a:3}_{b:3}\ng_{a:1}_{b:0}\ng_{a:0}_{b}\ng_{a}_{b:2}"
        )
        assert same(results[0], -f_)
        assert same(results[1], 1 / f_)
        assert same(results[2], r**2 * sin(theta) ** 2)
        assert results[3] == 0
        assert same(results[4], Array([-f_, 0, 0, 0]))
        assert str(results[4].indices) == "_{b}"
        assert same(results[5], Array([0, 0, r**2, 0]))

    def test_components_by_coordinate_name(self, schwarzschild):
        assert same(schwarzschild.expr("g_{t t}"), -f_)
        assert same(schwarzschild.expr(r"g_{\theta\theta}"), r**2)
        assert same(schwarzschild.expr("g_{a=r b=r}"), 1 / f_)
        assert same(schwarzschild.expr("g_{01}"), 0)

    def test_tensor_times_scalar(self, schwarzschild):
        x = Symbol("x")
        result = schwarzschild.expr("g_{a}_{b}*f(x)")
        assert same(result, SCHWARZSCHILD * Function("f")(x))
        assert str(result.indices) == "_{ab}"


class TestCurvature:
    def test_christoffel_symbols(self, schwarzschild):
        gamma = schwarzschild.expr("C^{a}_{b}_{c}")
        expected = spacetimes.schwarzschild(M, G=G).christoffel
        assert same(gamma, expected)
        assert str(gamma.indices) == "^{a}_{bc}"

    def test_christoffel_components(self, schwarzschild):
        results = schwarzschild.exe("C^{a:0}_{b:0}_{c:0}\nC^{a}_{b:0}_{c:0}\nC^{a:0}_{b}_{c:0}")
        assert results[0] == 0
        assert same(results[1], Array([0, G * M * (r - 2 * G * M) / r**3, 0, 0]))
        assert same(results[2], Array([0, G * M / (r * (r - 2 * G * M)), 0, 0]))

    def test_christoffel_by_coordinate_names(self, schwarzschild):
        assert same(schwarzschild.expr(r"\Gamma^{r}_{\theta\theta}"), 2 * G * M - r)
        assert same(schwarzschild.expr(r"\Gamma^{\phi}_{\theta\phi}"), cos(theta) / sin(theta))

    def test_christoffel_from_its_formula(self, schwarzschild):
        difference = schwarzschild.expr(
            "T^{a}_{c f} := (1/2)*g^{a}^{b}*(d_{c}*g_{b}_{f} + d_{f}*g_{b c} - d_{b}*g_{c f})"
            " - C^{a}_{c f}\n"
            "T^{a}_{c f}"
        )
        assert zero(difference)
        assert str(difference.indices) == "^{a}_{cf}"

    def test_riemann(self, schwarzschild):
        riemann = schwarzschild.expr("R^{a}_{b}_{c}_{h}")
        assert same(riemann, spacetimes.schwarzschild(M, G=G).riemann)
        assert str(riemann.indices) == "^{a}_{bch}"

    def test_riemann_components(self, schwarzschild):
        results = schwarzschild.exe(
            "R^{a:0}_{b:0}_{c:0}_{n:0}\nR^{a}_{b:0}_{c:0}_{n:0}\nR^{a:0}_{b:1}_{c:0}_{n}"
        )
        assert results[0] == 0
        assert same(results[1], Array([0, 0, 0, 0]))
        assert same(results[2], Array([0, 2 * G * M / (r**2 * (r - 2 * G * M)), 0, 0]))

    def test_riemann_from_its_formula(self, schwarzschild):
        difference = schwarzschild.expr(
            "R^{a}_{m}_{b}_{n} - ( d_{b}*C^{a}_{n m} + C^{a}_{b l}*C^{l}_{n m}"
            " - d_{n}*C^{a}_{b m} - C^{a}_{n l}*C^{l}_{b m} )"
        )
        assert zero(difference)
        assert str(difference.indices) == "^{a}_{mbn}"

    def test_definitions_reorder_indices(self, schwarzschild):
        result = schwarzschild.expr(
            "T^{a}_{m}_{b}_{n} := d_{b}*C^{a}_{n}_{m} + C^{a}_{b}_{l}*C^{l}_{n}_{m}"
            " - d_{n}*C^{a}_{b}_{m} - C^{a}_{n}_{l}*C^{l}_{b}_{m}\n"
            "T^{a}_{m b n} - R^{a}_{m b n}"
        )
        assert zero(result)

    def test_ricci_is_zero_in_vacuum(self, schwarzschild):
        ricci = schwarzschild.expr("Ric_{a}_{b}")
        assert ricci.components == Array.zeros(4, 4)
        assert str(ricci.indices) == "_{ab}"
        assert schwarzschild.expr("R_{\\mu\\nu}").components == Array.zeros(4, 4)

    def test_ricci_from_contracting_riemann(self, schwarzschild):
        ricci = schwarzschild.expr("R^{a}_{b}_{a}_{h}")
        assert ricci.components == Array.zeros(4, 4)
        assert str(ricci.indices) == "_{bh}"

    def test_ricci_scalar(self, schwarzschild):
        assert schwarzschild.expr("Ric") == 0
        assert schwarzschild.expr("R") == 0

    def test_einstein_tensor(self, schwarzschild):
        einstein = schwarzschild.expr("G_{a}_{b}")
        assert einstein.components == Array.zeros(4, 4)
        results = schwarzschild.exe(
            "Ric_{a b} - Ric*g_{a b}\n"
            "Ric_{a b} - g_{a b}*Ric\n"
            "Ric_{a b} - Ric*g_{a b}*Ric - G_{a b}"
        )
        for result in results:
            assert result.components == Array.zeros(4, 4)
            assert str(result.indices) == "_{ab}"

    def test_the_metric_is_covariantly_constant(self, schwarzschild):
        result = schwarzschild.expr("D_{a}*g_{b c}")
        assert zero(result)
        assert str(result.indices) == "_{abc}"
        assert zero(schwarzschild.expr("T_{b}_{a}_{c} := D_{a}*g_{b}_{c}\nT_{b}_{a}_{c}"))
        assert zero(schwarzschild.expr(r"\nabla_\lambda g^{\mu\nu}"))

    def test_kretschmann_scalar(self, schwarzschild):
        expected = 48 * G**2 * M**2 / r**6
        assert same(schwarzschild.expr("R_{abcd} R^{abcd}"), expected)
        assert same(schwarzschild.expr("Kretschmann"), expected)


class TestTensors:
    def test_components_from_lists(self, wb):
        x, y = symbols("x y")
        result = wb.expr("Coordinates := [x, y]\nT_{a} := [0, 0]\nT_{a}")
        assert result.components == Array([0, 0])
        assert result.coordinates == (x, y)

    def test_components_of_a_rank_two_tensor(self, wb):
        result = wb.expr(
            "Coordinates := [t, r, theta]\n"
            "T_{a}_{b} := [[1, 0, 0], [0, r**2, 0], [0, 0, r**2*sin(theta)**2]]\n"
            "T_{a}_{b}"
        )
        assert result.components == Array([[1, 0, 0], [0, r**2, 0], [0, 0, r**2 * sin(theta) ** 2]])

    def test_raising_and_lowering_round_trip(self, wb):
        result = wb.expr(
            "Coordinates := [t, r, theta, phi]\n"
            "g_{mu nu} := [[-(1 - (2 * G * M) / (c**2*r)), 0, 0, 0],"
            "[0, 1 / (1 - (2 * G * M) / (c**2*r)), 0, 0],[0, 0, r**2, 0],"
            "[0, 0, 0, r**2 * sin(theta) ** 2]]\n"
            "V_{a} := [1 + t , h(r, t), h(theta), h(r, t)]\n"
            "V_{a} - V^{b} * g_{a b}"
        )
        assert result.components == Array([0, 0, 0, 0])
        assert str(result.indices) == "_{a}"

    def test_symmetrisation(self, wb):
        wb.exe("Coordinates := [x, y]\nT_{ab} := [[1, 2], [3, 4]]")
        assert wb.expr("T_{(ab)}").components == Array(
            [[1, sympy.Rational(5, 2)], [sympy.Rational(5, 2), 4]]
        )
        assert wb.expr("T_{[ab]}").components == Array(
            [[0, -sympy.Rational(1, 2)], [sympy.Rational(1, 2), 0]]
        )

    def test_partial_derivative_indices(self, wb):
        x, y = symbols("x y")
        wb.exe("Coordinates := [x, y]\nV_{a} := [x^2 y, y^3]")
        assert wb.expr("V_{a,b}").components == Array([[2 * x * y, x**2], [0, 3 * y**2]])
        assert wb.expr("\\partial_b V_{a}").components == Array([[2 * x * y, 0], [x**2, 3 * y**2]])

    def test_kronecker_delta(self, wb):
        wb.exe("Coordinates := [x, y]\nV^{a} := [3, 4]")
        assert wb.expr("\\delta^{a}_{b} V^{b}").components == Array([3, 4])
        assert wb.expr("\\delta^{a}_{a}") == 2


class TestDerivations:
    def test_schwarzschild_from_a_spherically_symmetric_ansatz(self, wb):
        result = wb.expr(
            """
            Coordinates := [t, r, theta, phi]
            g_{mu}_{nu} := [[-A(r),0,0,0], [0,B(r),0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]]
            eq0 := Ric_{mu=0 nu=0}
            eq1 := Ric_{mu=1 nu=1}
            eq2 := Ric_{mu=2 nu=2}
            eq5 := (eq0*B(r) + eq1*A(r))*(r*B(r))
            B1 := RHS( dsolve(eq5, B(r)) )
            eq6 := simplify( subs(eq2, B(r), B1) )
            A1 := RHS( dsolve(eq6, A(r)) )
            g_{mu}_{nu} := [[A1,0,0,0], [0,1/A1,0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]]
            g_{mu}_{nu}
            """
        )
        C1, C2 = symbols("C1 C2")
        expected = Array(diag(C1 + C2 / r, 1 / (C1 + C2 / r), r**2, r**2 * sin(theta) ** 2))
        assert same(result, expected)

    def test_newtonian_limit_fixes_the_integration_constant(self, wb):
        result = wb.expr(
            r"""
            Coordinates := [t, r, theta, phi]
            g_{mu}_{nu} := [[1 + C/r, 0, 0, 0], [0, 1/(1 + C/r), 0, 0],
                            [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]]
            Gamma^{a}_{c f} := (1/2)*g^{a b}*(d_{c}*g_{b f} + d_{f}*g_{b c} - d_{b}*g_{c f})
            a1 := Gamma^{a:1}_{c:0 f:0}
            solve( \lim_{r -> oo} a1*r**2 + G*M , C)
            """
        )
        assert result == [-2 * G * M]

    def test_kretschmann_scalar_from_scratch(self, wb):
        result = wb.expr(
            """
            Coordinates := [t, r, theta, phi]
            g_{mu}_{nu} := [[-(1 - (2 * G * M) / (c**2*r)), 0, 0, 0],
                            [0, 1 / (1 - (2 * G * M) / (c**2*r)), 0, 0],
                            [0, 0, r**2, 0],
                            [0, 0, 0, r**2 * sin(theta) ** 2]]
            Gamma^{a}_{c f} := (1/2)*g^{a b}*(d_{c}*g_{b f} + d_{f}*g_{b c} - d_{b}*g_{c f})
            Riemann^{a}_{m b n} := d_{b}*Gamma^{a}_{n m} + Gamma^{a}_{b l}*Gamma^{l}_{n m} -
                                   d_{n}*Gamma^{a}_{b m} - Gamma^{a}_{n l}*Gamma^{l}_{b m}
            Ricci_{m}_{n} := Riemann^{a}_{m}_{a}_{n}
            TempOne^{a}^{f}^{h}^{s} := g^{s}^{q}*( g^{h}^{c}*(
                                           g^{f}^{b} * Riemann^{a}_{b}_{c}_{q} ) )
            TempTwo_{a}_{f}_{h}_{s} := g_{a}_{n}*Riemann^{n}_{f}_{h}_{s}
            S := TempOne^{a}^{f}^{h}^{s}*TempTwo_{a}_{f}_{h}_{s}
            simplify(S)
            """
        )
        assert result == 48 * G**2 * M**2 / (c**4 * r**6)  # exactly 48, not 48.0


class TestLatexRelativity:
    def test_line_element(self, wb):
        result = wb.expr(
            r"""
            Coordinates := [t, r, \theta, \phi]
            ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right) c^2 dt^2 +
                   \left(1 - \frac{2GM}{c^2 r}\right)^{-1} dr^2 +
                   r^2 \left(d\theta^2 + \sin^2\theta \, d\phi^2\right)
            g_{\mu\nu}
            """
        )
        assert same(result, spacetimes.schwarzschild(M, G=G, c=c).components)

    def test_line_element_with_solid_angle(self, wb):
        wb.exe(r"Coordinates := [t, r, \theta, \phi]" "\n" r"ds^2 = -dt^2 + dr^2 + r^2 d\Omega^2")
        assert same(wb.expr("g_{\\mu\\nu}"), Array(diag(-1, 1, r**2, r**2 * sin(theta) ** 2)))

    def test_field_equations_from_latex(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr(r"R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu} = 0")
        assert wb.expr(r"G_{\mu\nu} = 0")

    def test_spacetimes_bring_their_coordinates(self, wb):
        wb.exe(r"g_{\mu\nu} := Kerr(M, a)")
        assert wb.coordinates == (t, r, theta, phi)

    def test_friedmann_equations(self, wb):
        a, k = Function("a")(t), Symbol("k")
        einstein = wb.expr(r"g_{\mu\nu} := FLRW(k)" "\n" r"G^{t}{}_{t}")
        assert same(einstein, -3 * (k + diff(a, t) ** 2) / a**2)

    def test_vacuum_field_equations_are_component_equations(self, wb):
        wb.exe(r"""
            Coordinates := [t, r, \theta, \phi]
            g_{\mu\nu} := diag(-A(r), B(r), r^2, r^2 \sin^2\theta)
        """)
        equations = wb.expr(r"R_{\mu\nu} = 0")
        assert isinstance(equations, TensorEquation)
        assert len(equations.equations()) == 4  # tt, rr, θθ and φφ (sin²θ times θθ)
        assert not equations

    def test_second_bianchi_identity(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr(r"\nabla_{[\lambda} R_{\rho\sigma]\mu\nu} = 0")

    def test_a_covariant_derivative_sees_the_whole_tensor(self, wb):
        # The t component of a divergence needs every component of T, not just T^{μt}.
        wb.exe(
            r"""
            Coordinates := [t, r, \theta, \phi]
            ds^2 = -dt^2 + a(t)^2 (\frac{dr^2}{1 - k r^2} + r^2 d\Omega^2)
            u^{\mu} := [1, 0, 0, 0]
            T^{\mu\nu} := (\rho(t) + p(t)) u^{\mu} u^{\nu} + p(t) g^{\mu\nu}
            """
        )
        a, rho, p = (Function(n)(t) for n in ("a", "rho", "p"))
        continuity = diff(rho, t) + 3 * diff(a, t) / a * (rho + p)
        assert same(wb.expr(r"\nabla_\mu T^{\mu t}"), continuity)
        assert same(wb.expr(r"T^{\mu t}{}_{;\mu}"), continuity)
        assert same(wb.expr(r"\nabla_\mu T^{\mu\nu}")[0], continuity)
        # Taken twice, the components are still chosen last, so each ∇ sees whole tensors.
        assert same(
            wb.expr(r"\nabla_r \nabla_\mu T^{\mu t}"),
            wb.expr(r"\nabla_\lambda \nabla_\mu T^{\mu\nu}")[1, 0],
        )

    def test_a_covariant_derivative_refuses_components_chosen_inside(self, wb):
        wb.exe(
            r"""
            Coordinates := [t, r, \theta, \phi]
            g_{\mu\nu} := Schwarzschild(M)
            A^{\mu} := [0, f(r), 0, 0]
            """
        )
        for source in (r"\nabla_\mu (A^{\mu} A^{t})", r"\nabla_\mu (g_{tt} A^{\mu})"):
            with pytest.raises(IndexNotationError, match="components chosen inside its operand"):
                wb.expr(source)
        # A component named first is a function like any other, and a power is not a
        # component. Here the divergence of a radial vector is ∂_r (r² V^r) / r².
        f, h = Function("f")(r), 2 * M / r - 1
        wb.exe("h := g_{tt}")
        assert same(wb.expr(r"\nabla_\mu (h A^{\mu})"), diff(r**2 * h * f, r) / r**2)
        assert same(wb.expr(r"\nabla_\mu (r^{2} A^{\mu})"), diff(r**4 * f, r) / r**2)

    def test_conservation_of_the_einstein_tensor(self, wb):
        wb.exe(r"g_{\mu\nu} := ReissnerNordstrom(M, Q)")
        assert zero(wb.expr(r"\nabla_\mu G^{\mu\nu}"))

    def test_euler_lagrange_equations_give_the_geodesic_equations(self, wb):
        wb.exe(r"""
            Coordinates := [t, r, \theta, \phi]
            g_{\mu\nu} := Schwarzschild(M)
            x^{\mu} := [t(\tau), r(\tau), \theta(\tau), \phi(\tau)]
        """)
        tau = Symbol("tau")
        lagrangian = wb.expr(
            r"L := -g_{\mu\nu} \dot{x}^\mu \dot{x}^\nu"
            r" |_{t = t(\tau), r = r(\tau), \theta = \theta(\tau), \phi = \phi(\tau)}"
            "\n"
            r"\frac{d}{d\tau} \frac{\partial L}{\partial \dot{x}^\mu}"
            r" - \frac{\partial L}{\partial x^\mu}"
        )
        assert lagrangian.rank == 1
        # the θ equation: d/dτ(2 r² θ') - 2 r² sin θ cos θ φ'² = 0
        R, Th, Ph = (Function(n)(tau) for n in ("r", "theta", "phi"))
        expected = -(
            2 * diff(R**2 * diff(Th, tau), tau) - 2 * R**2 * sin(Th) * cos(Th) * diff(Ph, tau) ** 2
        )
        assert same(lagrangian[2], -expected) or same(lagrangian[2], expected)

    def test_geodesic_equations(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        equations = wb.expr(r"geodesic_equations(\tau)")
        assert len(equations) == 4
