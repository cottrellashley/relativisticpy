r"""SciPy's optimisation.

Linear programs, and those whose variables are whole numbers, by ``scipy.optimize.linprog``
and ``scipy.optimize.milp``; convex quadratic programs, by the linear system of their KKT
conditions at each set of active constraints; and any other optimum without parameters by
``scipy.optimize.minimize``, from several starting points, or, of a whole number, over the
reals and then at the whole numbers either side. A problem written with decimals, as data
is, comes here first; one SymPy finds no closed form for, after.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any

import sympy
from sympy import Symbol

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, Found, Problem, optimizer

x, y, w = sympy.symbols("x y w", real=True)
m, k = sympy.symbols("m k", integer=True)
n = sympy.Symbol("n", integer=True, nonnegative=True)

#: The most inequalities whose sets of active ones a quadratic program is solved at.
_MOST_ACTIVE = 12
#: The most whole-number variables whose neighbours are tried, after the reals.
_MOST_WHOLE = 6


def _coefficients(expr: Any, variables: Sequence[Symbol]) -> tuple[list[float], float]:
    """A linear expression as the coefficient of each variable, and a constant."""
    expanded = sympy.expand(expr)
    coefficients = [float(expanded.coeff(v)) for v in variables]
    constant = float(expanded.xreplace(dict.fromkeys(variables, 0)))
    return coefficients, constant


def _number(value: Any) -> Any:
    """A number SciPy found, as SymPy writes it: a whole number where it is one."""
    value = float(value)
    return sympy.Integer(round(value)) if abs(value - round(value)) < 1e-9 else sympy.Float(value)


@optimizer(
    "linprog",
    summary="linear programs, by SciPy's HiGHS: linprog, or milp where the variables are "
    "whole numbers",
    requires="scipy",
    numerical=True,
    examples=[
        Example(r"\max_{x, y} \; 0.5x + 0.3y \quad \text{subject to} \quad x + y \leq 10, \quad "
                r"x \leq 6, \quad x, y \geq 0", 4.2,
                "scipy.optimize.linprog([-0.5, -0.3], A_ub=[[1, 1], [1, 0]], b_ub=[10, 6])"),
        Example(r"\max_{x, y \in \mathbb{Z}} \; x + y \quad \text{subject to} \quad "
                r"2x + 2y \leq 7, \quad x, y \geq 0", 3,
                "scipy.optimize.milp([-1, -1], constraints=LinearConstraint([[2, 2]], ub=7), "
                "integrality=[1, 1])"),
    ],
    problems=[
        Problem("max", 3 * x + 5 * y, (x, y),
                inequalities=(4 - x, 12 - 2 * y, 18 - 3 * x - 2 * y, x, y)),
        Problem("min", x + y, (x, y), inequalities=(x + 2 * y - 4, 3 * x + y - 6, x, y)),
        Problem("max", m + k, (m, k), inequalities=(7 - 2 * m - 2 * k, m, k)),
    ],
)  # fmt: skip
class LinearProgram:
    def takes(self, problem: Problem) -> bool:
        degree = problem.degree()
        return (
            degree is not None
            and degree <= 1
            and not problem.parameters
            and not problem.strict
            and not problem.distinct
            and all(set(values) <= {0, 1} for _, values in problem.finite)
        )

    def optimum(self, problem: Problem) -> Found | None:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, linprog, milp

        variables = problem.variables
        sign = -1.0 if problem.sense == "max" else 1.0
        objective, constant = _coefficients(problem.objective, variables)
        cost = sign * np.asarray(objective)
        rows, lower, upper = [], [], []
        for g in problem.inequalities:  # g >= 0
            coefficients, rest = _coefficients(g, variables)
            rows.append(coefficients)
            lower.append(-rest)
            upper.append(np.inf)
        for h in problem.equalities:
            coefficients, rest = _coefficients(h, variables)
            rows.append(coefficients)
            lower.append(-rest)
            upper.append(-rest)
        binary = {v for v, values in problem.finite}
        whole = [v.is_integer or v in binary for v in variables]
        low = [0.0 if v in binary else -np.inf for v in variables]
        high = [1.0 if v in binary else np.inf for v in variables]
        if any(whole):
            constraints = [LinearConstraint(np.asarray(rows), lower, upper)] if rows else []
            result = milp(
                cost,
                constraints=constraints,
                integrality=np.asarray(whole, dtype=int),
                bounds=Bounds(low, high),
            )
            how = "by SciPy's milp (HiGHS)"
        else:
            ub = [(i, r) for i, r in enumerate(rows) if math.isinf(upper[i])]
            eq = [(i, r) for i, r in enumerate(rows) if not math.isinf(upper[i])]
            result = linprog(
                cost,
                A_ub=np.asarray([-np.asarray(r) for _, r in ub]) if ub else None,
                b_ub=np.asarray([-lower[i] for i, _ in ub]) if ub else None,
                A_eq=np.asarray([r for _, r in eq]) if eq else None,
                b_eq=np.asarray([lower[i] for i, _ in eq]) if eq else None,
                bounds=list(zip(low, high, strict=True)),
                method="highs",
            )
            how = "by SciPy's linprog (HiGHS)"
        if result.status == 2:
            raise EvaluationError(f"no point keeps the constraints of {problem}")
        if result.status == 3:
            infinite = sympy.oo if problem.sense == "max" else -sympy.oo
            return Found(infinite, None, how, proven=True, attained=False, numeric=True)
        if result.status != 0 or result.x is None:
            return None
        point = tuple(_number(v) for v in result.x)
        value = _number(sign * result.fun + constant)
        return Found(value, point, how, proven=True, numeric=True)


def _quadratic(problem: Problem) -> tuple[Any, Any, float] | None:
    """A convex quadratic objective to make least as its Hessian, its gradient at 0 and
    its value there, f = x^T H x / 2 + c^T x + d; ``None`` where it is no such thing."""
    import numpy as np

    sign = 1 if problem.sense == "min" else -1
    f = sign * sympy.sympify(problem.objective)
    variables = list(problem.variables)
    hessian = sympy.hessian(f, variables)
    if hessian.free_symbols or any(sympy.diff(f, v, 3) != 0 for v in variables):
        return None
    at_zero = dict.fromkeys(variables, 0)
    gradient = [sympy.diff(f, v).xreplace(at_zero) for v in variables]
    try:
        matrix = np.asarray(hessian.tolist(), dtype=float)
        vector = np.asarray(gradient, dtype=float)
        constant = float(f.xreplace(at_zero))
    except (TypeError, ValueError):
        return None
    if np.linalg.eigvalsh((matrix + matrix.T) / 2).min() < -1e-12:
        return None
    return matrix, vector, constant


@optimizer(
    "quadratic",
    summary="convex quadratic programs: the linear system of the KKT conditions at each "
    "set of active constraints, by SciPy's linear algebra",
    requires="scipy",
    numerical=True,
    examples=[
        Example(r"\arg\min_{w_1, w_2} \; 0.04 w_1^2 + 0.09 w_2^2 \quad \text{subject to} \quad "
                r"w_1 + w_2 = 1, \quad w_1, w_2 \geq 0", [0.09 / 0.13, 0.04 / 0.13],
                "standard: the least variance of two assets, w_1 = s_2^2/(s_1^2 + s_2^2)"),
        Example(r"\min_{x, y} \; x^2 + y^2 - 1.2 x \quad \text{subject to} \quad x + y \geq 1",
                -0.28, "scipy.optimize.minimize(..., method='SLSQP').fun, at x = 0.8, y = 0.2"),
    ],
    problems=[
        Problem("min", (x - 1) ** 2 + (y - 2) ** 2, (x, y), inequalities=(1 - x - y,)),
        Problem("min", x**2 + 2 * y**2, (x, y), inequalities=(x + y - 1, x, y)),
        Problem("max", -(x**2) - y**2 + x, (x, y), equalities=(x - y,)),
    ],
)  # fmt: skip
class QuadraticProgram:
    def takes(self, problem: Problem) -> bool:
        degree = problem.degree()
        if degree is None or degree > 2 or problem.parameters or not problem.continuous():
            return False
        if problem.strict or problem.distinct or len(problem.inequalities) > _MOST_ACTIVE:
            return False
        linear = all(
            sympy.Poly(g, *problem.variables).total_degree() <= 1
            for g in (*problem.equalities, *problem.inequalities)
            if sympy.sympify(g).has(*problem.variables)
        )
        return linear and _quadratic(problem) is not None

    def optimum(self, problem: Problem) -> Found | None:
        import numpy as np
        from scipy.linalg import lstsq

        found = _quadratic(problem)
        if found is None:
            return None
        hessian, gradient, constant = found
        variables = problem.variables
        size = len(variables)
        equalities = [_coefficients(h, variables) for h in problem.equalities]
        inequalities = [_coefficients(g, variables) for g in problem.inequalities]
        best: tuple[float, Any] | None = None
        for count in range(min(len(inequalities), size) + 1):
            for active in itertools.combinations(range(len(inequalities)), count):
                rows = [c for c, _ in equalities] + [inequalities[i][0] for i in active]
                rests = [r for _, r in equalities] + [inequalities[i][1] for i in active]
                held = np.asarray(rows, dtype=float).reshape(len(rows), size)
                system = np.block([[hessian, -held.T], [held, np.zeros((len(rows), len(rows)))]])
                right = np.concatenate([-gradient, -np.asarray(rests, dtype=float)])
                solution, *_ = lstsq(system, right)
                point, multipliers = solution[:size], solution[size:]
                if np.linalg.norm(system @ solution - right) > 1e-9 * max(
                    1.0, np.linalg.norm(right)
                ):
                    continue
                if np.any(multipliers[len(equalities) :] < -1e-10):
                    continue
                if any(np.dot(c, point) + r < -1e-9 for c, r in inequalities):
                    continue
                value = 0.5 * point @ hessian @ point + gradient @ point + constant
                if best is None or value < best[0] - 1e-12:
                    best = (value, point)
        if best is None:
            return None
        sign = 1 if problem.sense == "min" else -1
        point = tuple(_number(v) for v in best[1])
        how = "by SciPy's linear algebra, from the KKT conditions at each set of active constraints"
        return Found(_number(sign * best[0]), point, how, proven=True, numeric=True)


def _numeric(expr: Any, variables: Sequence[Symbol]) -> Callable[..., Any]:
    """An expression of the variables as a function of a vector of their values."""
    function = sympy.lambdify([list(variables)], expr, modules=["scipy", "numpy"])
    return lambda values: float(function(list(values)))


def _starts(problem: Problem, size: int) -> list[Any]:
    """Where the search starts: the middle of each variable's range, or 0, then 1, then a
    few points drawn with a fixed seed, so that a result can be found again."""
    import numpy as np

    ranges = problem.ranges()
    middle = []
    for v in problem.variables:
        low, high = ranges[v]
        if low.is_finite and high.is_finite:
            middle.append((float(low) + float(high)) / 2)
        elif low.is_finite:
            middle.append(float(low) + 1.0)
        elif high.is_finite:
            middle.append(float(high) - 1.0)
        else:
            middle.append(0.0)
    generator = np.random.default_rng(0)
    drawn = [np.asarray(middle) + generator.normal(scale=2.0, size=size) for _ in range(4)]
    return [np.asarray(middle), np.ones(size), *drawn]


@optimizer(
    "minimize",
    summary="any other optimum without parameters, by SciPy's minimize (SLSQP) from several "
    "starting points; of whole numbers, over the reals, then at the whole numbers either "
    "side",
    requires="scipy",
    numerical=True,
    examples=[
        Example(r"\min_{x} \; \cosh x + x^4 - 3x", -0.653190237988869,
                "scipy.optimize.minimize_scalar(lambda x: np.cosh(x) + x**4 - 3*x).fun",
                tolerance=1e-9),
        Example(r"\arg\max_{n \in \mathbb{N}} \frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2} + 1\right)}",
                5, "standard: the volume of the unit ball is greatest in five dimensions"),
    ],
    problems=[
        Problem("min", (x - 1) ** 2 + (y - 2) ** 2, (x, y), inequalities=(1 - x - y,)),
        Problem("min", x + y, (x, y), equalities=(x**2 + y**2 - 2,)),
        Problem("max", x * sympy.exp(-x), (x,), inequalities=(x,)),
    ],
)  # fmt: skip
class Minimize:
    def takes(self, problem: Problem) -> bool:
        whole = [v for v in problem.variables if v.is_integer]
        return (
            not problem.parameters
            and not problem.finite
            and len(whole) <= _MOST_WHOLE
            and bool(problem.variables)
        )

    def optimum(self, problem: Problem) -> Found | None:
        import numpy as np
        from scipy.optimize import minimize

        variables = problem.variables
        size = len(variables)
        sign = 1.0 if problem.sense == "min" else -1.0
        objective = _numeric(sign * problem.objective, variables)
        gradient_of = [sympy.diff(sign * problem.objective, v) for v in variables]
        try:
            gradients = sympy.lambdify([list(variables)], gradient_of, modules=["scipy", "numpy"])
            jacobian: Any = lambda values: np.asarray(gradients(list(values)), dtype=float)  # noqa: E731
        except (TypeError, ValueError, NameError):
            jacobian = None
        constraints = (
            [{"type": "ineq", "fun": _numeric(g, variables)} for g in problem.inequalities]
            + [
                {"type": "ineq", "fun": _numeric(g - sympy.Float(1e-12), variables)}
                for g in problem.strict
            ]
            + [{"type": "eq", "fun": _numeric(h, variables)} for h in problem.equalities]
        )
        ranges = problem.ranges()
        bounds = [
            (float(low) if low.is_finite else None, float(high) if high.is_finite else None)
            for low, high in (ranges[v] for v in variables)
        ]
        best: tuple[float, Any] | None = None
        starts = _starts(problem, size)
        for start in starts:
            try:
                with np.errstate(all="ignore"):
                    result = minimize(
                        objective,
                        np.clip(
                            start,
                            [b[0] if b[0] is not None else -np.inf for b in bounds],
                            [b[1] if b[1] is not None else np.inf for b in bounds],
                        ),
                        jac=jacobian,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                        options={"maxiter": 500, "ftol": 1e-14},
                    )
            except (ValueError, TypeError, OverflowError, ZeroDivisionError):
                continue
            if not np.all(np.isfinite(result.x)) or not math.isfinite(float(result.fun)):
                continue
            if problem.violations([sympy.Float(v) for v in result.x], 1e-7, whole=False):
                continue
            if best is None or result.fun < best[0] - 1e-12:
                best = (float(result.fun), np.asarray(result.x))
        if best is None:
            return None
        how = f"by SciPy's minimize (SLSQP), from {len(starts)} starting points"
        point = [float(v) for v in best[1]]
        whole = [k for k, v in enumerate(variables) if v.is_integer]
        if whole:
            return self._whole_numbers(problem, point, whole, how)
        value = sign * best[0]
        return Found(
            sympy.Float(value),
            tuple(sympy.Float(v) for v in point),
            how,
            proven=problem.convex(),
            numeric=True,
        )

    def _whole_numbers(
        self, problem: Problem, point: list[float], whole: list[int], how: str
    ) -> Found | None:
        """Of whole-number variables: the best of the whole numbers either side of the
        optimum over the reals, with the others as they were found."""
        choices = [
            sorted({math.floor(point[k]), math.ceil(point[k])}) if k in whole else [point[k]]
            for k in range(len(point))
        ]
        found: list[tuple[Any, tuple[Any, ...]]] = []
        for candidate in itertools.product(*choices):
            exact = tuple(
                sympy.Integer(v) if k in whole else sympy.Float(v) for k, v in enumerate(candidate)
            )
            if problem.violations(exact, 1e-7):
                continue
            found.append((problem.at(exact), exact))
        if not found:
            return None
        greatest = problem.sense == "max"
        value, chosen = (max if greatest else min)(found, key=lambda pair: float(pair[0]))
        how += ", over the reals and then at the whole numbers either side"
        return Found(value, chosen, how, proven=False, numeric=True)
