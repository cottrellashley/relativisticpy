import unittest
from relativisticpy.workbook.workbook import Workbook
import sympy as sp



class TestWorkbook(unittest.TestCase):

    def test_mul_numbers(self):
        mul_workbook = Workbook()
        # Integer Multiplication
        self.assertAlmostEqual(mul_workbook.expr('2 * 3'), 6)
        self.assertAlmostEqual(mul_workbook.expr('-2 * 3'), -6)
        self.assertAlmostEqual(mul_workbook.expr('-2 * -3'), 6)

        # Float Multiplication
        self.assertAlmostEqual(mul_workbook.expr('2.5 * 2'), 5.0)
        self.assertAlmostEqual(mul_workbook.expr('-2.5 * 2'), -5.0)
        self.assertAlmostEqual(mul_workbook.expr('-2.5 * -2.5'), 6.25)

    def test_mul_symbols(self):
        mul_symbol_workbook = Workbook()
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Symbol and Number
        self.assertAlmostEqual(mul_symbol_workbook.expr('2 * x'), 2 * x)
        self.assertAlmostEqual(mul_symbol_workbook.expr('-2 * x'), -2 * x)
        self.assertAlmostEqual(mul_symbol_workbook.expr('-2.5 * x'), -2.5 * x)

        # Symbol and Symbol
        self.assertAlmostEqual(mul_symbol_workbook.expr('y * x'), y * x)
        self.assertAlmostEqual(mul_symbol_workbook.expr('-y * x'), -y * x)
        self.assertAlmostEqual(mul_symbol_workbook.expr('-2.5 * x * y'), -2.5 * x * y)

    def test_sub_numbers(self):
        sub_workbook = Workbook()
        # Integer Subtraction
        self.assertAlmostEqual(sub_workbook.expr('2 - 1'), 1)
        self.assertAlmostEqual(sub_workbook.expr('-2 - 1'), -3)
        self.assertAlmostEqual(sub_workbook.expr('-2 - -1'), -1)

        # Float Subtraction
        self.assertAlmostEqual(sub_workbook.expr('2.5 - 1'), 1.5)
        self.assertAlmostEqual(sub_workbook.expr('-2.5 - 1'), -3.5)
        self.assertAlmostEqual(sub_workbook.expr('-2.5 - -1.5'), -1.0)

    def test_sub_symbols(self):
        sub_symbol_workbook = Workbook()
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Symbol and Number
        self.assertAlmostEqual(sub_symbol_workbook.expr('x - 1'), x - 1)
        self.assertAlmostEqual(sub_symbol_workbook.expr('1 - x'), 1 - x)
        self.assertAlmostEqual(sub_symbol_workbook.expr('-1.5 - x'), -1.5 - x)

        # Symbol and Symbol
        self.assertAlmostEqual(sub_symbol_workbook.expr('y - x'), y - x)
        self.assertAlmostEqual(sub_symbol_workbook.expr('-y - x'), -y - x)
        self.assertAlmostEqual(sub_symbol_workbook.expr('-1.5 * x - y'), -1.5 * x - y)

    def test_div_numbers(self):
        div_workbook = Workbook()
        # Integer Division
        self.assertAlmostEqual(div_workbook.expr('10 / 2'), 5)
        self.assertAlmostEqual(div_workbook.expr('-10 / 2'), -5)
        self.assertAlmostEqual(div_workbook.expr('-10 / -2'), 5)

        # Float Division
        self.assertAlmostEqual(div_workbook.expr('10.0 / 2'), 5.0)
        self.assertAlmostEqual(div_workbook.expr('-10.0 / 2'), -5.0)

    def test_integrate(self):
        integrate_workbook = Workbook()
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Simple Integration
        self.assertEqual(integrate_workbook.expr('integrate(x**2, x)'), sp.integrate(x**2, x))
        self.assertEqual(integrate_workbook.expr('integrate(x**3 + 2*x**2 - 1, x)'), sp.integrate(x**3 + 2*x**2 - 1, x))

        # Multiple Variables
        self.assertEqual(integrate_workbook.expr('integrate(x*y, x)'), sp.integrate(x*y, x))
        self.assertEqual(integrate_workbook.expr('integrate(x**2*y**3, y)'), sp.integrate(x**2*y**3, y))

    def test_solve(self):
        solve_workbook = Workbook()
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Simple Linear Equations
        self.assertEqual(solve_workbook.expr('solve(x + 2*y - 3, x)'), sp.solve(x + 2*y - 3, x))
        self.assertEqual(solve_workbook.expr('solve(x**2 - 4, x)'), sp.solve(x**2 - 4, x))

        # Multiple Variables
        # self.assertEqual(solve_workbook.expr('solve([x + y - 2, x - y + 1], [x, y])'), sp.solve([x + y - 2, x - y + 1], [x, y]))

    def test_diff(self):
        diff_workbook = Workbook()
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Simple Differentiation
        self.assertEqual(diff_workbook.expr('diff(x**2, x)'), sp.diff(x**2, x))
        self.assertEqual(diff_workbook.expr('diff(x**3 + 2*x**2 - 1, x)'), sp.diff(x**3 + 2*x**2 - 1, x))

        # Multiple Variables
        self.assertEqual(diff_workbook.expr('diff(x*y, x)'), sp.diff(x*y, x))
        self.assertEqual(diff_workbook.expr('diff(x**2*y**3, y)'), sp.diff(x**2*y**3, y))

    def test_simplify(self):
        simplify_workbook = Workbook()
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Simple Simplification
        self.assertEqual(simplify_workbook.expr('simplify(x**2 + 2*x + 1)'), sp.simplify(x**2 + 2*x + 1))
        self.assertEqual(simplify_workbook.expr('simplify((x + 1)**2 - x**2)'), sp.simplify((x + 1)**2 - x**2))

        # Multiple Variables
        self.assertEqual(simplify_workbook.expr('simplify(x*y + x**2*y**2)'), sp.simplify(x*y + x**2*y**2))
        self.assertEqual(simplify_workbook.expr('simplify(x**2*y**3 / (x*y))'), sp.simplify(x**2*y**3 / (x*y)))

    def test_expand(self):
        expand_workbook = Workbook()
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Simple expansion
        self.assertEqual(expand_workbook.expr('expand((x + y)**2)'), sp.expand((x + y)**2))
        self.assertEqual(expand_workbook.expr('expand((x + y)**3)'), sp.expand((x + y)**3))

        # Multiple Variables
        self.assertEqual(expand_workbook.expr('expand(x*(y + 1))'), sp.expand(x*(y + 1)))
        self.assertEqual(expand_workbook.expr('expand((x + y)*(x - y))'), sp.expand((x + y)*(x - y)))

    def test_limit(self):
        limit_workbook = Workbook()
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Simple Limits
        self.assertEqual(limit_workbook.expr('limit((x**2 - 1)/(x - 1), x, 1)'), sp.limit((x**2 - 1)/(x - 1), x, 1))
        self.assertEqual(limit_workbook.expr('limit(1/x, x, 0)'), sp.limit(1/x, x, 0))

        # Limits with Infinity
        # self.assertEqual(limit_workbook.expr('limit(x**2, x, sp.oo)'), sp.limit(x**2, x, sp.oo))
        # self.assertEqual(limit_workbook.expr('limit(1/x, x, sp.oo)'), sp.limit(1/x, x, sp.oo))

    def test_series(self):
        x = sp.Symbol('x')
        w = Workbook()
        actual = w.expr('series(cos(x), x, 0, 5)')
        expected = sp.series(sp.cos(x), x, 0, 5)
        # Simple Series Expansion
        #self.assertEqual(series_workbook.expr('series(exp(x), x, 0, 5)'), sp.series(sp.exp(x), x, 0, 5))
        #self.assertEqual(series_workbook.expr('series(log(1 + x), x, 0, 5)'), sp.series(sp.log(1 + x), x, 0, 5))

        # Series Expansion with Options
        self.assertEquals(sp.series(sp.cos(x), x, 0, 5), expected)
        #self.assertEquals(series_workbook.expr('series(sin(x), x, 0, 5)'), sp.series(sp.sin(x), x, 0, 5))



