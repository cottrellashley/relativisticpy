# RelativisticPy

**RelativisticPy reads mathematics as it is written, in LaTeX, plain text or Unicode,
and computes it.** Its core is algebra, calculus, linear algebra and index notation over
any space. Probability, signals, optimisation, number theory, quantum mechanics and
General Relativity are read in the notation of their books. It is for people, in Python,
at the command line and in notebooks, and for AI agents, as a tool.

> **For:** everyone. **You will learn:** what RelativisticPy is, and where to start for
> what you want to do.

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"\int_0^\infty x^2 e^{-x} \, dx")
2
>>> wb.expr("d/dx sin(x)^2")
2*sin(x)*cos(x)
>>> wb.expr("∑_{n=1}^∞ 1/n²")
pi**2/6
>>> wb.expr(r"\det \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}")
5
>>> wb.exe(r"X \sim \mathcal{N}(0, 1)")
[]
>>> wb.expr(r"\Pr(X > 1)")
1/2 - erf(sqrt(2)/2)/2
>>> wb.expr(r"ds^2 = -\left(1 - \frac{2M}{r}\right) dt^2 + \frac{dr^2}{1 - \frac{2M}{r}} + r^2 d\Omega^2")
>>> wb.expr(r"R_{abcd} R^{abcd}")
48*M**2/r**6

```

## Why

Computer algebra asks you to translate mathematics into its own functions first:
`integrate(x**2*exp(-x), (x, 0, oo))` for \(\int_0^\infty x^2 e^{-x}\,dx\).
RelativisticPy reads the notation itself, as books, papers and people write it, so an
equation copied from a paper runs as it is, and a whole paper can be
[checked](checking.md), equation by equation. Index notation never needed spacetime: an
index can range over coordinates, assets, days or the states of a qubit.

Notation is ambiguous, and a wrong answer is worse than none. Where RelativisticPy
assumes what a book would, a note says so. What it cannot read or decide, it refuses,
saying why and what to write instead. A wrong answer given without a word is a bug, and
the [roadmap](roadmap.md) says which are known.

## Where to start

Install it and run a first program in [Getting started](getting-started.md). Then,
by what you want to do:

| I want to | Read, in order |
|---|---|
| compute: algebra, calculus, matrices, probability and more | [The language](language.md), the page of [each field](language.md#fields), [LaTeX](latex.md) |
| give an AI agent mathematics to do | [MCP server, for agents](mcp.md), [the language reference](reference.md) |
| study General Relativity | [Tensors and geometry](tensors.md), [Solving tensor equations](notation.md), [Numerics](numerics.md) |
| check a paper | [LaTeX](latex.md), [Checking papers](checking.md) |
| compute with data | [Beyond spacetime](spaces.md), [Python API](python.md) |
| know the language | [The language](language.md), [the language reference](reference.md), [Glossary](glossary.md) |
| work in Jupyter or marimo | [Jupyter and notebooks](jupyter.md), [Live mathematics in notebooks](live.md), [Examples](examples.md) |
| move from another tool | [Coming from other tools](coming-from.md), [Coming from 1.x](upgrading.md) |
| know how it works | [How it works](architecture.md), [Roadmap](roadmap.md) |

When a result surprises you, see [Troubleshooting](troubleshooting.md).

## For agents

`relativisticpy mcp` serves RelativisticPy to an agent over the Model Context Protocol:
tools to compute, to verify a step and to check a paper ([MCP server](mcp.md)). Each value
says whether it is a complete answer, and a refusal says why.

Everything here is in the installed package too: `relativisticpy help` lists the
pages, `relativisticpy help reference` prints the whole language on one page, and
`relativisticpy help NAME` one function in full, as `relativisticpy help besselj`. The
site serves [llms.txt](https://relativisticpy.org/llms.txt), an index of these pages for
language models, and [llms-full.txt](https://relativisticpy.org/llms-full.txt), all of
them in one file.
