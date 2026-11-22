# RelativisticPy

**RelativisticPy reads mathematics as it is written, in LaTeX, plain text or Unicode,
and computes it.** Type a line element and get its curvature; write index notation
over NumPy or pandas data; run `relativisticpy check paper.tex` to see which of a
paper's equations hold.

> **For:** everyone. **You will learn:** what RelativisticPy is, and where to start for
> what you want to do.

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"ds^2 = -\left(1 - \frac{2M}{r}\right) dt^2 + \frac{dr^2}{1 - \frac{2M}{r}} + r^2 d\Omega^2")
>>> wb.expr(r"\Gamma^{r}_{tt}")
M*(-2*M + r)/r**3
>>> wb.expr(r"G_{\mu\nu} = 0")
G_{μν} = 0: holds in all 10 components
>>> wb.expr(r"R_{abcd} R^{abcd}")
48*M**2/r**6

```

## Why

Calculations in General Relativity are long, mechanical and easy to get wrong by hand.
Computer algebra systems can do them, but each asks you to translate your equations
into its own vocabulary of functions and objects first.

RelativisticPy asks nothing of the kind. Its input *is* the notation of the field:
indices up and down, Einstein summation, Christoffel symbols written as
`\Gamma^\lambda{}_{\mu\nu}`, derivatives written as `\partial_\mu` or `\nabla_\mu`. It
reads quick plain text and LaTeX alike, so an equation copied from a paper or from your
notes runs as it is, and a whole paper can be [checked](checking.md), equation by
equation. Index notation never needed spacetime, so the same notation computes with data
too: an index can range over assets, days or the states of a qubit.

## Where to start

Install it and run a first program in [Getting started](getting-started.md). Then,
by what you want to do:

| I want to | Read, in order |
|---|---|
| study General Relativity | [Tensors and geometry](tensors.md), [Solving tensor equations](notation.md), [Numerics](numerics.md) |
| check a paper | [LaTeX](latex.md), [Checking papers](checking.md) |
| compute with data | [Beyond spacetime](spaces.md), [Python API](python.md) |
| know the language | [The language](language.md), [the language reference](reference.md), [Glossary](glossary.md) |
| work in Jupyter | [Jupyter and notebooks](jupyter.md), [Examples](examples.md) |
| move from another tool | [Coming from other tools](coming-from.md), [Upgrading from 1.x](upgrading.md) |
| know how it works | [How it works](architecture.md), [Roadmap](roadmap.md) |

When a result surprises you, see [Troubleshooting](troubleshooting.md).

## For agents

Everything here is in the installed package too: `relativisticpy help` lists the
pages, and `relativisticpy help reference` prints the whole language on one page. The
site serves [llms.txt](https://relativisticpy.org/llms.txt), an index of these pages for
language models, and [llms-full.txt](https://relativisticpy.org/llms-full.txt), all of
them in one file.
