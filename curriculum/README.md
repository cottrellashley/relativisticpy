# The curriculum

What RelativisticPy should read, across the mathematical sciences, and how far it has got.
Every case is a short program written as a textbook or paper writes it, with an answer
known without RelativisticPy. Cases are grouped by domain and graded by level, so the
curriculum documents what is easy and what is hard, and the cases that fail say what to
build next.

```bash
uv run python benchmarks/curriculum.py                # the scoreboard
uv run python benchmarks/curriculum.py --failures     # why each case fails
uv run python benchmarks/curriculum.py --update       # record the cases that pass now
uv run python benchmarks/curriculum.py --seeds 0,1,2  # the cases the hash seed changes
```

`passing.txt` lists the cases that pass. A case may join it, never leave it:
`tests/test_curriculum.py` runs every case listed there.

A run fixes Python's hash seed, starting again with `PYTHONHASHSEED=0` as the command
line does, so that the scoreboard is the same every time; the tests run under the same
seed in continuous integration. `--seeds` runs the curriculum once under each seed given,
each in a process of its own, and lists the cases whose outcome differs between them,
with each run's reason: findings, which do not fail the run. Continuous integration runs
it under seeds 0, 1 and 2, as warnings on the cases.

## Levels

| Level | Name | What it asks of the language | Examples |
|---|---|---|---|
| 1 | notation | one expression, written as a textbook prints it, with an immediate answer | `\frac{d}{dx} \sin x`, `\int_0^1 x^2\,dx`, `\binom{5}{2}` |
| 2 | computation | one standard computation from a definition: a student's exercise | the Christoffel symbols of a sphere, the eigenvalues of \(\sigma_x\), a Taylor series |
| 3 | derivation | several statements building on each other, as on a page of a textbook | the Friedmann equations from the FLRW metric, the ladder operators of the oscillator |
| 4 | specialist | the notation of research papers and specialist texts | covariant derivatives in index notation, bra–ket algebra, differential forms, slashed momenta |
| 5 | frontier | what no design of the language covers yet, written down so the gap is known | path integrals, Feynman diagrams, renormalisation |

The level is about what the language must read and do, not how hard the physics is. A
famous result can be level 2, and an obscure notation level 4.

## Domains

Each domain is a folder, and each topic within it a Markdown file. Examples:

- `general-relativity/`;
- `differential-geometry/`;
- `special-relativity/`;
- `electromagnetism/`;
- `classical-mechanics/`;
- `quantum-mechanics/`;
- `quantum-field-theory/`;
- `analysis/`;
- `complex-analysis/`;
- `linear-algebra/`;
- `differential-equations/`;
- `vector-calculus/`;
- `fourier-analysis/`;
- `probability-and-statistics/`;
- `statistical-mechanics/`;
- `data/`;
- `numerical-methods/`;
- `optimization/`;
- `signal-processing/`;
- `machine-learning/`;
- `graphs-and-networks/`;
- `dynamical-systems-and-control/`;
- `economics-and-finance/`;
- `number-theory-and-combinatorics/`;
- `group-theory/`;
- `fluid-dynamics/`;
- `thermodynamics/`;
- `astrophysics-and-cosmology/`;
- `geometry-and-topology/`;
- `chemistry/`.

## A case

A topic file starts with `# Title` and a paragraph on the topic. Each case is a `##`
section with, in this order:

1. **A line naming its level, its identifier and its source:**
   `**Level 2** · `gr.schwarzschild.kretschmann` · Carroll (2004), eq. (5.33)`.
   - The identifier is unique: the domain's short name, the topic, then the case.
   - The source is where the result can be checked: a book, a paper, or "standard".
2. **A sentence or two saying what the case asks of the language.**
3. **The program,** in a fenced block marked `latex`, or `relativisticpy` for plain
   text. Write it as the book or paper writes it, not as the language reads it today: a
   case that fails is a finding, not a mistake.
4. **The oracle,** in a fenced block marked `python`. It computes the answer with SymPy,
   NumPy or SciPy, never with RelativisticPy. SymPy's names are in scope, and NumPy as
   `np`. It sets one of:
   - `expected`: the value of the program's last output. Possible values:
     - a number or SymPy expression;
     - `True` or `False`, for an equation that must hold or must not;
     - a list or SymPy array of every component of a tensor;
     - a dictionary from components, named by coordinates (`("t", "r")`), to values;
     - the solutions of a `solve`: the values of one unknown (`[2/3]`), tuples of values in
       the order the program names its unknowns (`[(8, sqrt(3))]` for
       `solve(..., [K, \omega])`), or mappings from unknowns to values. The language may
       answer in any of these forms, and the solutions in any order;
     - `None`, for a program that shows nothing.
   - `outputs`: a list with the value of every output, in order.
   - `error`: the name of the error the program must raise, such as
     `"IndexNotationError"`, for notation that is wrong and must be refused. `message`
     may give a part of its text.

   `tolerance` sets the relative tolerance of numbers (default `1e-9`). Symbols are
   compared by name, whatever is assumed of them.

````markdown
## The Kretschmann scalar of Schwarzschild

**Level 2** · `gr.schwarzschild.kretschmann` · Carroll (2004), eq. (5.33)

The square of the Riemann tensor, contracted over every index pair.

```latex
g_{\mu\nu} := Schwarzschild(M)
R_{\alpha\beta\gamma\delta} R^{\alpha\beta\gamma\delta}
```

```python
M, r = symbols("M r")
expected = 48 * M**2 / r**6
```
````

## Rules for oracles

- **Know the answer.** An oracle computes it independently, and agrees with the cited
  source. Where a book gives a result, the oracle reproduces that result, not merely
  something plausible.
- **Keep oracles short and plain:** SymPy for exact answers, NumPy or SciPy for numbers.
  An oracle never imports RelativisticPy.
- **Never fit a case to the program.** Write the program as people write the
  mathematics. If the language cannot read it yet, the case fails, and that is the point.
- **One idea per case.** A case that fails should say one thing about what to build.

## The notation matrix

`notations/` holds the same mathematics written every way a book writes it. A construct
of the core, such as a fraction, a derivative or a definite integral, names its **forms**,
its **operands** and its **oracle**; a **cell** is one form filled with one operand,
`calculus.derivative/leibniz-fraction/x^3`. Every form of a construct must give the answer
its oracle gives, so two spellings of the same mathematics that disagree are found by
running it.

```bash
uv run python benchmarks/notations.py                        # the scoreboard, in seconds
uv run python benchmarks/notations.py --construct calculus   # some constructs
uv run python benchmarks/notations.py --failures             # why each cell fails
uv run python benchmarks/notations.py --show CELL            # a cell, as a case
uv run python benchmarks/notations.py --oracles              # each oracle, by two routes
uv run python benchmarks/notations.py --update               # record what passes now
uv run python benchmarks/notations.py --seeds 0,1,2          # what the hash seed changes
```

It reports by construct, by notation and by form: pass, wrong, refused. It lists every
wrong answer with its program, the answer given and the answer expected, and, for each
operand, the forms that disagree with each other. A cell is run as a case is, in a fresh
workbook, within the same time limit, its answer compared as strictly; each runs in a
process of its own, so that none depends on the cells before it.
`notations/passing.txt` records the cells that pass, and `tests/test_notations.py` keeps
them passing.

A construct is a Markdown file, `notations/<area>/<construct>.md`, named after its
folder and its file. Markdown, because a person reads it on the page and extends it by
copying a form; a table for the operands, because tens of them are compared at a glance.
It starts with `# Title` and a paragraph that names its books, then has these sections:

- **`## Names`:** Python, with SymPy's names in scope, that defines what the operands and
  the oracle are written with: `x = symbols("x")`.
- **`## Operands`:** a table. The first column names an operand, on the first of its
  rows; the second says which notations a row is written in, of `latex`, `text`,
  `unicode` and `python`; each column headed by a slot, as `` `f` ``, says how that slot
  is written in them; a last column says what misreading the operand separates. A bar in
  a cell is written `\|`. An operand not written in a notation has no cell of its forms.
- **`## Oracle`:** Python that computes the answer from the operand's slots alone, as
  `python` writes them, and sets `expected`, as a case's oracle does. It never imports
  RelativisticPy, and never sees the form.
- **`## Second route`:** Python run after the oracle, which computes the answer another
  way and asserts that the two agree.
- **`## Setup`,** if needed: the lines every form's program starts with, a fenced block
  for each notation, and a `python-scope` block for Python objects a program reads.
- **`## Forms`:** a `###` section for each form, with a line naming it and its source,
  ``**Form** · `leibniz-fraction` · Stewart (2015), §2.8``, and its template in a fenced
  block marked `latex`, `text` or `unicode`, where `«f»` stands for the slot `f`. A
  template is written as its source writes it, never as the language reads it: a cell
  that fails is a finding. Two more lines are optional:
  ``**Not of** · `sum` · why``, or ``**Only of** · ...``, where a source does not write
  the form of every operand, and ``**Refused** · `ParseError` · where it is decided``,
  for a form that a decision says is refused: its cells pass when the program is refused
  with that error, and an answer is a failure.

A program in LaTeX is given to the language as LaTeX, in every statement
(`design/notation.md`, decision 10); one in plain text or Unicode is given as it is typed.

````markdown
## Oracle

```python
expected = diff(f, var)
```

## Forms

### Leibniz's fraction, of a function's name

**Form** · `leibniz-fraction` · Stewart (2015), §2.8, Other Notations: `\frac{dy}{dx}`

```latex
«name»(«var») := «f»
\frac{d«name»}{d«var»}
```
````
