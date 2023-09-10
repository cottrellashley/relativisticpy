from relativisticpy.gr import Connection, Ricci, Riemann
from relativisticpy.core import Indices, Idx, MultiIndexObject, Metric, Mathify
from relativisticpy.workbook.workbook import Mathify_two, Workbook

if __name__ == '__main__':

    # basis = Mathify('[t, r]')

    # test = Metric.from_string('_{a}_{b}','[[1,r],[0,r**2]]', '[r, theta]')

    # idcs = Indices(-Idx('a'), Idx('b'), Idx('c'), Idx('d'))

    # rie = Riemann(idcs, test, basis)

    # print(rie[Indices(Idx('a'), Idx('b'), Idx('c'), Idx('d'))])

    wb = Workbook()

    print(wb.expr('exp(r) - t + w '))

    # print(wb.expr('x + y '))

    # Test cases: The following is the list of equations to derive the Schild Black Hole Solution 

    # Write a string exoression with all the nodes we defined:
    define_symbol = 'MetricSymbol := G ' # Works
    define_coordinates = 'Coordinates := [t, r, theta, phi]'
    exp2 = 'G_{mu}_{nu} := [[-A_{a:1}_{r:2},0,0,0], [0,B(r),0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]]' # Works
    exp3 = 'MetricBasis := [t, r, theta, phi]' # Works
    exp4 = 'a = Ric_{mu:0}_{nu:0}' # Works
    exp5 = 'eq5 = ((a*B + b*A)*(r*B))' # Works
    exp6 = 'B_solution = diff_solve(eq5, B)'
    exp7 = 'eq6 = subs(c, B, C_0/A)'
