from relativisticpy.gr import Connection, Ricci, Riemann
from relativisticpy.core import Indices, Idx, MultiIndexObject, Metric, Mathify
from relativisticpy.workbook.workbook import Workbook
import re

if __name__ == '__main__':

    wb = Workbook()

    # print(wb.exe('/Users/ashleycottrell/code/repositories/relativisticpy/.example-files/black_hole.txt'))

    # print(wb.exe('.example-files/basic_calculus_example.txt'))

    # print(wb.exe('/Users/ashleycottrell/code/repositories/relativisticpy/.example-files/Schild_solution.txt'))

    print(wb.expr("""
    # First: Define the pre-requisites of our problem

    # Symbols we want Metric and Ricci to have
    MetricSymbol := G
    RicciSymbol := Ric

    Coordinates := [t, r, theta, phi]

    # Define the metric tensor components: we define a general spherically symmetric tensor
    G_{mu}_{nu} := [[-A(r),0,0,0], [0,B(r),0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]]

    # Now we have defined the metric above, we can call any individual component of the Ricci tensor itself (as it is metric dependent)
    eq0 = Ric_{mu:0}_{nu:0}
    eq1 = Ric_{mu:1}_{nu:1}
    eq2 = Ric_{mu:2}_{nu:2}

    # We construct a simplified equation from the Ricci components as follows
    eq5 = (eq0*B(r) + eq1*A(r))*(r*B(r))

    # We call the dsolve method which is a differential equation solver
    solutionB = dsolve(eq5, B(r)) # returns B(r) = C1/A(r)

    # We know the solution of B(r) = C1/A(r) so we substitute it in the third Ricci equation we have available to us
    eq6 = simplify(subs(eq2, B(r), C1/A(r)))

    # Finally set the answers as A and B
    A = dsolve(eq6, A(r))

    B = 1/A

    # Output the answer as latex, if you wish: (remove latex method if you do not)
    [[-A,0,0,0], [0,B,0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]]

    """))