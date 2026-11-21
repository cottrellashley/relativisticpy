# Examples

Each Markdown file here is a notebook. Its `$$...$$` blocks are a RelativisticPy
program, and GitHub typesets them as the equations they are. Run one, or see
its results typeset under each step:

```bash
relativisticpy examples/schwarzschild.md
relativisticpy examples/schwarzschild.md --render
```

| Example | |
|---|---|
| [schwarzschild.md](schwarzschild.md) | The Schwarzschild metric, derived from Einstein's equations in vacuum |
| [kretschmann.md](kretschmann.md) | The Kretschmann scalar by hand, index by index, checked against the built-in |
| [friedmann.md](friedmann.md) | The Friedmann equations and the continuity equation of an expanding universe |
| [geodesics.md](geodesics.md) | The geodesic equations from a Lagrangian, checked against the Christoffel symbols |
| [light.py](light.py) | Python: light bent and captured by a black hole, integrated numerically |
| [spins.md](spins.md) | Two spins: an operator on a product space, its exact spectrum, and the evolution of a spin |
| [portfolio.py](portfolio.py) | Python: 500 assets, the tangency portfolio, risk as a metric, and real factors told from noise |

Every example runs as part of the test suite, in `tests/test_examples.py`.
