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
    MetricSymbol := G 
    RicciSymbol := Ric 
    Coordinates := [t, r, theta, phi] 
    G_{mu}_{nu} := [[-A(r),0,0,0], [0,B(r),0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]] 
    eq0 = Ric_{mu:0}_{nu:0} 
    eq1 = Ric_{mu:1}_{nu:1} 
    eq2 = Ric_{mu:2}_{nu:2} 
    eq5 = (eq0*B(r) + eq1*A(r))*(r*B(r)) 
    solutionB = dsolve(eq5, B(r)) 
    eq6 = simplify(subs(eq2, B(r), C1/A(r))) 
    A = dsolve(eq6, A(r)) 
    B = 1/A 
    [[-A,0,0,0], [0,B,0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]] 
    """))