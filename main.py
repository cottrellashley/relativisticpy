from relativisticpy.gr import Connection, Ricci, Riemann
from relativisticpy.core import Indices, Idx, MultiIndexObject, Metric, Mathify
from relativisticpy.workbook.workbook import Workbook
import re

if __name__ == '__main__':

    wb = Workbook()

    # print(wb.exe('/Users/ashleycottrell/code/repositories/relativisticpy/.example-files/black_hole.txt'))

    # print(wb.exe('/Users/ashleycottrell/code/repositories/relativisticpy/.example-files/basic_calculus_example.txt'))

    print(wb.exe('/Users/ashleycottrell/code/repositories/relativisticpy/.example-files/Schild_solution.txt'))

    # print(wb.expr('numerical(pi + e, 20) ** 5.8598744820488384738 '))