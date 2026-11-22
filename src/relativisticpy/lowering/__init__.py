"""The lowerer: the IR of each statement, emitted from what the reader and the typer say.

The lowerer walks a statement's syntax and emits the closed IR of named operations
(:mod:`relativisticpy.ir`), computing nothing. At each name and each ambiguous form it
asks the reader what it means (:mod:`relativisticpy.reading`), and at each index the typer
whether it keeps the rules (:mod:`relativisticpy.typing`). Its parts are layers, each
lowering one kind of thing and using those below it:

- :mod:`~relativisticpy.lowering.notation`: abstract index notation, and its components;
- :mod:`~relativisticpy.lowering.tensors`: tensors with indices, and their derivatives;
- :mod:`~relativisticpy.lowering.statistics`: statistics over an index;
- :mod:`~relativisticpy.lowering.linalg`: linear algebra on tensors;
- :mod:`~relativisticpy.lowering.expressions`: the walk of an expression;
- :mod:`~relativisticpy.lowering.definitions`: definitions;
- :mod:`~relativisticpy.lowering.transforms`: transforms, ranges of indices and sums over
  data;
- :mod:`~relativisticpy.lowering.optima`: optima, over an index of data or over variables;
- :mod:`~relativisticpy.lowering.objects`: the Python objects a program names, as
  ``\\texttt{prices}``, read first of all.
"""
