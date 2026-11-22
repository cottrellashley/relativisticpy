# The curriculum

What RelativisticPy should read, across the mathematical sciences, and how far it has got.
Every case is a short program written as a textbook or paper writes it, with an answer
known without RelativisticPy. Cases are grouped by domain and graded by level, so the
curriculum documents what is easy and what is hard, and the cases that fail say what to
build next.

```bash
uv run python benchmarks/curriculum.py              # the scoreboard
uv run python benchmarks/curriculum.py --failures   # why each case fails
uv run python benchmarks/curriculum.py --update     # record the cases that pass now
```

`passing.txt` lists the cases that pass. A case may join it, never leave it:
`tests/test_curriculum.py` runs every case listed there.

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
- `thermodynamics/`.

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
