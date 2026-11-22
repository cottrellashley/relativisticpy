# How it works

> **For:** contributors, and the curious. **You will learn:** how text becomes results,
> stage by stage, and where to add a function, a command, a spacetime or an operation.

```text
source text ─▶ lexer ─▶ tokens ─▶ parser ─▶ syntax tree ─▶ compiler ─▶ IR graph ─▶ executor ─▶ results
```

Each stage has one job, its own tests, and a module of its own:

| Stage | Module | Job |
|---|---|---|
| Lexer | `relativisticpy.language.lexer` | text to tokens; plain text, LaTeX and Unicode become one stream |
| Parser | `relativisticpy.language.parser` | tokens to a syntax tree that records what was written |
| Reader | `relativisticpy.reading` | what each name and each ambiguous form means, with a note where it could surprise |
| Typer | `relativisticpy.typing` | the rules of index notation and spaces: free and summed indices, positions, spaces |
| Lowerer | `relativisticpy.lowering` | the IR of each statement, a closed graph of named operations, computing nothing |
| Compiler | `relativisticpy.compiler` | the three passes over a statement, the session and the scope |
| IR | `relativisticpy.ir` | a graph of typed operations, interned |
| Executor | `relativisticpy.runtime` | computes the values outputs need, cached, in parallel |
| Algebra | `relativisticpy.tensors`, `relativisticpy.diffgeom` | tensors, contraction, curvature, simplification |

You can watch the pipeline at work:

```python
>>> from relativisticpy import Workbook
>>> from relativisticpy.language import parse, to_sexpr, tokenize
>>> [token.value for token in tokenize(r"\Gamma^{\mu}_{\nu\rho}")][:6]
['Gamma', '^', '{', 'mu', '}', '_']
>>> to_sexpr(parse(r"\frac{1}{2} g^{ab}(\partial_c g_{bd})"))
'(* (* (/ 1 2) (g ^[a b])) (* (∂ _[c]) (g _[b d])))'
>>> print(Workbook().ir(r"""
...     Coordinates := [x, y]
...     g_{ab} := diag(1, x^2)
...     \Gamma^{x}_{yy}
... """))
%0 = const 1                                             Scalar
%1 = const x                                             Scalar
%2 = const 2                                             Scalar
%3 = pow(%1, %2)                                         Scalar
%4 = call diag(%0, %3)                                   Value
%5 = to-array (2, 2)(%4)                                 Value
%6 = metric(%5)                                          Tensor_{ab}
%8 = inverse(%6)                                         Tensor^{ab}
%11 = christoffel(%6, %8)                                Tensor^{#v0}_{#v1 #v2}
%12 = select 0=0, 1=1, 2=1(%11)                          Scalar
out %12

```

## The lexer

The lexer is where the two dialects become one. Greek letters in every
spelling (`\theta`, `theta`, `θ`, `\vartheta`) become the identifier `theta`;
operator commands become operators (`\cdot` is `*`, `\leq` is `<=`,
`\coloneqq` is `:=`); `\left`/`\right` pairs are matched, with display
brackets becoming grouping parentheses; and everything typographic is
absorbed: spacing, sizing, labels, math delimiters, alignment markers. Each
token records whether whitespace preceded it, whether that included explicit
LaTeX spacing such as `\,`, and whether it was written inside `\mathrm{...}`.
The LaTeX and Unicode vocabulary is data, in `relativisticpy.language.vocabulary`.

## The parser

A Pratt parser with one table of binding powers. It recognises notation, not
meaning: implicit multiplication, scripts and index lists, every derivative
notation, integrals that end at their differential, big operators, matrices,
`cases`, evaluation bars, function definitions and blocks.

Where notation is ambiguous, the tree keeps every reading. A script such as
`^{a}` is stored both as an expression (a power) and as an index list (a
component), and the compiler, which knows whether the base is a tensor,
chooses. When a script cannot be an index list, the reason is kept too, so
that if the compiler needs indices it can say precisely why they are wrong.

Errors point at the source and suggest fixes, and `to_sexpr` prints any tree
as an S-expression, which is how the parser's tests state what notation means.

## The compiler

The compiler gives the tree meaning, using a symbol table of what each name
is: a variable, a function, a tensor (with its index positions), the metric,
or nothing yet. It reads a statement in three passes, each a package with one job:

- the *reader*, `relativisticpy.reading`, decides what each name and each ambiguous
  form means, and says so in a note where it takes the less likely reading;
- the *typer*, `relativisticpy.typing`, holds the rules of index notation and spaces,
  each with one message;
- the *lowerer*, `relativisticpy.lowering`, emits the IR: notation, tensors and their
  derivatives, statistics, linear algebra, expressions and definitions, each a module.

They are one walk of the syntax tree, as reading a form may need what was made of the
forms before it: the lowerer walks, and asks the reader and the typer. The `Compiler`
in `relativisticpy.compiler` is the three together, and orchestrates statements, with
the session and the scope. The reader records each decision it takes, with why, as the
statement's *reading*: `Workbook.explain` shows the readings of a program, computing
nothing.

The reader decides

- whether `x^{a}` is a power or a component, and whether `r_{s}` names a symbol;
- which built-in tensor a name means, by name and number of indices (`R` with
  four indices is Riemann, with two Ricci, with none the Ricci scalar);
- whether `=` defines (`y = 2`, `ds^2 = ...`, a new tensor) or states an equation;
- what a run of letters means: one name in plain text, a product in LaTeX;
- what an operator such as `\partial_\mu` or `\frac{d}{dx}` acts on.

It checks index notation before anything is computed, so a mistake costs
milliseconds rather than a Riemann tensor. Every value has a static type,
`Scalar`, `Tensor[indices]`, `Equation` or `Value`, and the index rules are
checked on these types: free indices of a sum must agree, an index may appear
at most twice, a summed index once up and once down.

The compiler computes nothing. What it needs to know comes from what is written or
from a declaration: a spacetime declares the coordinates its metric is in, a space
written `\mathbb{R}^{3}` its dimension, and a number of eigenvalues is written as one.
A definition that needs something of the value it makes, as a metric must be one, makes
a *check*: a node of the IR that the workbook computes as it runs the statement, before
the next is compiled. A metric that is degenerate, of the wrong size or not symmetric
is so refused where it is defined, saying why, and the definition is undone.

Tensor notation is lowered to a small core of operations. Raising or lowering
an index is a contraction with the metric; a covariant derivative is a partial
derivative plus one contraction with the Christoffel symbols per index; a trace
is a contraction of a tensor with itself; `T_{(ab)}` is a symmetrisation.
User functions are inlined at their call sites, in a child of the scope they
were defined in, which gives lexical scoping.

A tensor without components, such as \(T_{\mu\nu}\) in Einstein's equations,
is kept as notation instead: operations of the IR, `notation.tensor`, `notation.add`,
`notation.mul`, `notation.solve`, `notation.subs`, `notation.trace`,
`notation.simplify` and the rest, which the engines compute as SymPy tensor
expressions, from `relativisticpy.abstract`, through canonical forms computed by the
Butler–Portugal algorithm, with the identities of General Relativity added. What the
compiler needs to know of notation comes from each operation's type rule, in
`relativisticpy.typing.notation`: whether it is an equation, and the free indices of
each side, as they are written. Notation is computed as the statement that writes it
runs, as a check is, in the order it is written, so that what cannot be computed, as a
solve for a tensor the equation does not contain, is refused there. Each scalar factor
of notation, as \(8\pi G\), is an operation of the IR too; one computed from tensors is
refused.

Where notation meets components, as when a tensor is defined from it with a metric in
force, its terms are known only once it is computed. So the lowering is staged:
`notation.lower`, as the statement runs, has the lowerer write each term in the
language, its scalar coefficient bound to a fresh name, and compile it, in the
coordinates and with the metric and tensors in force where it is written; and
`notation.components` is the value of that IR. Its type comes from its rule: the free
indices of the notation, in the order they are written, over their spaces. This is the
one type that the computed value could have told otherwise: notation that vanishes has
zero components, with the free indices it is written with.

Every index ranges over a space, and a tensor's type records the space of each
axis: spacetime in General Relativity, or assets, days, a qubit (see
[Beyond spacetime](spaces.md)). An index label takes its space from a declaration,
`a, b \in \mathcal{A}`, or from the first tensor it is written on in a statement;
labels that do neither range over spacetime. The rules of summation follow each
space's metric: where it is the identity, an index may be summed in any position.
Expectations bind their index, so that factors share it, and are contracted with the
measure of its space in the same contraction as the factors. Products with an inverse
become linear solves.

## The IR

The IR is a graph of nodes, each an operation applied to other nodes, with a
static type. Operations are frozen data: `Contract`, `Permute`, `Select`,
`Partial`, `Symmetrize`, `Map`, `Add`, `Mul`, `Scale`, `Pow`, `Call`, `MakeArray`,
`ToArray`, `Equation`, `Inverse`, `LineElement` and `Geometry`.

No operation holds a Python function. A `Call`, or a `Map` over every component of a
tensor, holds the name of what it computes and its attributes, values known when
compiling. The registry of implementations, `relativisticpy/runtime/registry.py`, maps
each name to the function that computes it, with how many arguments it takes, whether it
applies to each component of a tensor and what kind of value it gives. A builtin has the
name the language calls it by, as `sin` or `solve`; what the compiler emits for notation
has a name of its own, as `core.sum` for a sum, `core.subs` for an evaluation bar,
`linalg.solve` for an inverse times a vector or `notation.solve` for a solve in index
notation. So the whole program can be read, printed and compared before anything is
computed.

The same operations compute symbolic components with SymPy and numerical ones with
NumPy: the kernels look at what they are given. A contraction of numbers is
`numpy.einsum`, in the order its optimiser finds cheapest, in slices when its
operands are too large to hold at once, as a matrix product for a SciPy sparse
matrix, and with `jax.numpy` for JAX arrays, through which `Workbook.grad`
differentiates.

The graph interns nodes: asking for the same operation on the same arguments
returns the existing node, so common subexpressions are shared. Each node also
has a *computation key* that ignores its type. Index names are part of the
type, so `A_{ab}B^{b}{}_{c}` and `A_{xy}B^{y}{}_{z}` are different nodes with
different types, but they share one key and one computed value. Renaming
indices is therefore free.

## The executor

The workbook runs a program a statement at a time: it compiles the statement, has
the executor compute its checks, and then the values it shows. The executor computes a
node by first computing whatever it depends on, and remembers every value by
computation key. Nothing is computed that no output or check needs, and nothing is
computed twice: asking for the Ricci tensor and then the Einstein tensor computes the
Ricci tensor once.

Component-wise work, such as simplifying every component of a tensor or the
curvature kernels' simplification passes, goes through the executor's `map`.
With `Workbook(workers=N)`, large batches are spread over a pool of `N`
processes; SymPy expressions travel between them by pickling.

## Contraction

`relativisticpy.tensors.arrays.contract` is a sparse Einstein summation.
Curvature tensors are mostly zeros, so it multiplies only nonzero components,
and it contracts operands pairwise in a greedy order that keeps intermediate
results small. The Kretschmann scalar \(R_{abcd} R^{abcd}\), six operands with
the metric contractions, is a few hundred multiplications rather than the
\(4^8\) a dense product would need.

## Simplification

Components are kept in a normal form that suits relativity: trigonometric
functions are written with `sin` and `cos` only, and multiple angles are
expanded. This is how papers write Christoffel symbols (\(\cot\theta\) appears as
\(\cos\theta/\sin\theta\)), and it makes identities decidable by algebra.

Deciding whether a component vanishes is the most common question, so it has a
fast, exact answer: combine the expression into one fraction, expand the
numerator, and use \(\cos^2 x = 1 - \sin^2 x\) until no cosine has a power above
one. The numerator is then zero exactly when the component vanishes. This never
computes a polynomial GCD, the step that makes general simplification crawl.
It is why verifying that the Kerr metric is a vacuum solution takes seconds
rather than tens of minutes.

## Extending RelativisticPy

- **A function, or anything a library computes**: write an integration, a module of
  `relativisticpy/integrations/` that declares capabilities, each checked by the
  conformance suite; [Integrations](integrations.md) says how, step by step. The registry
  in `relativisticpy/runtime/registry.py` registers each capability under its names.
- **A kind of Python object**, read by name as `\texttt{prices}`: an object integration
  of the same package says what it becomes, data, a number, an expression or a function;
  [Integrations](integrations.md#python-objects) says how.
- **A LaTeX command or symbol**: add it to `relativisticpy/language/vocabulary.py`;
  structural commands need a case in the parser's `_command`.
- **A spacetime**: add a function to `relativisticpy/spacetimes.py`, declaring the
  coordinates its metric is in, which the compiler reads without computing it, and
  declare it as a constructor in `relativisticpy/integrations/spacetimes.py`.
- **An operation**: register what computes it in `relativisticpy/runtime/registry.py`
  and have the lowerer, in `relativisticpy/lowering/`, emit a `Call` of its name; one
  that is not a function of values, as a contraction is, is an op in
  `relativisticpy/ir/ops.py` with a kernel in `relativisticpy/runtime/kernels.py`. A
  rule of index notation it keeps goes in the typer, `relativisticpy/typing/`, and a new
  reading of what is written in the reader, `relativisticpy/reading/`.

[The language reference](reference.md) is written from these tables, and from the
declarations of the integrations, by `relativisticpy/reference.py`, which says what each
name means. A new command needs a line there too; the tests say which, and
`uv run python -m relativisticpy.reference --write docs/reference.md` writes the page
again.

Every stage has its own tests under `tests/`, and every example in these pages
runs as part of the test suite.

**Next:** the [roadmap](roadmap.md), for what comes next.
