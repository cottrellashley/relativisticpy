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
| Compiler | `relativisticpy.compiler` | the tree to a typed IR: meaning, index checks, lowering |
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
%7 = inverse(%6)                                         Tensor^{ab}
%10 = christoffel(%6, %7)                                Tensor^{#v0}_{#v1 #v2}
%11 = select 0=0, 1=1, 2=1(%10)                          Scalar
out %11

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
or nothing yet. It decides

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

Tensor notation is lowered to a small core of operations. Raising or lowering
an index is a contraction with the metric; a covariant derivative is a partial
derivative plus one contraction with the Christoffel symbols per index; a trace
is a contraction of a tensor with itself; `T_{(ab)}` is a symmetrisation.
User functions are inlined at their call sites, in a child of the scope they
were defined in, which gives lexical scoping.

A tensor without components, such as \(T_{\mu\nu}\) in Einstein's equations,
is kept as notation instead: a constant of the graph holding a SymPy tensor
expression, from `relativisticpy.abstract`. `solve`, `trace`, `simplify` and
`subs` work on it as notation, through canonical forms computed by the
Butler–Portugal algorithm, with the identities of General Relativity added.
Where notation meets components, as when a tensor is defined from it with a
metric in force, each of its terms is written back in the language, its scalar
coefficient bound to a fresh name, and compiled like any other tensor
expression.

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

The executor computes a node by first computing whatever it depends on, and
remembers every value by computation key. Nothing is computed that no output
needs, and nothing is computed twice: asking for the Ricci tensor and then the
Einstein tensor computes the Ricci tensor once.

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

- **A function**: add it to a table in `relativisticpy/runtime/functions.py`, from
  which `relativisticpy/runtime/registry.py` registers it under its name, marking
  whether it works component-wise on tensors.
- **A LaTeX command or symbol**: add it to `relativisticpy/language/vocabulary.py`;
  structural commands need a case in the parser's `_command`.
- **A spacetime**: add a function to `relativisticpy/spacetimes.py` and list it with
  the spacetimes in `relativisticpy/runtime/functions.py`.
- **An operation**: add an op to `relativisticpy/ir/ops.py`, a kernel to
  `relativisticpy/runtime/kernels.py`, and have the compiler emit it.

[The language reference](reference.md) is written from these tables by
`relativisticpy/reference.py`, which says what each name means. A new function,
command or spacetime needs a line there too; the tests say which, and
`uv run python -m relativisticpy.reference --write docs/reference.md` writes the page
again.

Every stage has its own tests under `tests/`, and every example in these pages
runs as part of the test suite.

**Next:** the [roadmap](roadmap.md), for what comes next.
