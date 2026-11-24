# The language reference

> **For:** anyone, person or agent, who needs to know what the language accepts. **You will learn:** every statement, command, environment, constant, verdict and error, and every function by name.

This page is generated from the tables the language is read with and from what its integrations declare. `relativisticpy help reference` prints it offline, and `relativisticpy help NAME` one function in full, as `relativisticpy help besselj`: its forms, its arguments, its examples with their answers, and what computes it. The guides explain each part: [The language](language.md) and the page of [each field](language.md#fields), [LaTeX](latex.md), [Tensors and geometry](tensors.md) and [Beyond spacetime](spaces.md).

## Statements

A program is a sequence of statements, one to a line; a semicolon, `\\` and `\linebreak \newline` also end one. `#` and `%` start a comment, so `%` is not a remainder. A statement goes on to the next line while a bracket is open, after a line that ends with an operator, and before a line that starts with one such as `+` or `=`; a line that starts with `-` goes on only if it is indented further.

`:=` defines, as do `\coloneq` `\coloneqq` `\defeq` `\equiv` `\triangleq` and `≡` `≔`; between two sides neither of which can be defined, `\equiv` states that they are equal. `=` defines only what has no meaning yet; otherwise it states an equation, or a claim, which is checked.

| Statement | Meaning |
|---|---|
| `x := 2` | defines x, or gives it a new value |
| `x = 2`, where x is new | defines x; so do `f(x) = x^2` and a new tensor `T_{ab} = ...` |
| `x = 3`, where x has a value | a claim, checked: does x equal 3? `x := 3` gives x a new value |
| `ds^2 = ...` | a line element: the metric, and the coordinates if none are declared |
| `R = 0`, `g_{tt} = -1` | with a metric in force, a claim about a built-in or a component |
| `t = 0`, where t is a coordinate | a condition: read, not computed |
| `x^2 + y^2 = 1` | an equation, to `solve`; `a = b = c` is a claim for each link |
| `f(x) := x^2` | a function; `f : (x) -> { ... }` is one with a block of statements |
| `\mathcal{H} := \mathbb{C}^{2}` | a space, complex, of 2 dimensions; `\mathbb{R}^{n}` is real |
| `i, j \in \mathcal{H}` | index labels that range over a space |
| `x_n := n h, \quad n = 0, \ldots, N-1` | a grid: n ranges over its N values, and x is made for each |
| `a > 0`, `0 < x < 1`, `a, b > 0`, `x \in \mathbb{R}`, `n \in \mathbb{N}` | an assumption about names nothing defines, from then on |
| `Coordinates := [t, r, \theta, \phi]` | the coordinates, in order |
| `Constants` `Reals` `Integers` `Complex` `Positive` `:= [a, b]` | declare a and b constants, real, integers, complex or positive, before they are used |

A renaming statement gives a built-in another name: `MetricSymbol := G` makes the metric `G`. They are, with the names they change: `MetricSymbol` (`g`), `ConnectionSymbol` (`C`), `RiemannSymbol` (`R`), `RicciSymbol` (`Ric`), `EinsteinSymbol` (`G`), `EinsteinTensorSymbol` (`G`), `WeylSymbol` (`W`), `DerivativeSymbol` (`d`), `CovariantDerivativeSymbol` (`D`).

## Names and numbers

- A name that nothing defines is a symbol. `theta`, `\theta` and `θ` are one symbol, as is every spelling of a Greek letter.
- In LaTeX, a run of letters is a product of one symbol for each letter: `\frac{2GM}{r}` is 2·G·M/r. Greek names, defined names and builtins stay whole, and `\mathrm{GM}` or `\text{GM}` is one name. In plain text, `GM` is one name, with a note where its letters are written alone too.
- Plain text and Unicode write LaTeX's functions and operators in letters, as AsciiMath does, and read as the commands they spell: `sin 2x` and `sin²x` are `\sin 2x` and `\sin^2 x`, and `log_2 8`, `int_0^1 x^2 dx`, `lim_(x->0) f(x)` and `sum_(n=1)^oo 1/n^2` are `\log_2 8`, `\int_0^1 x^2\,dx`, `\lim_{x \to 0} f(x)` and `\sum_{n=1}^{\infty} 1/n^2`; `oo` is `\infty`. A solidus between differentials is Leibniz's fraction: `d/dx f`, `dy/dx`, `d(x^3)/dx`, `∂f/∂x`. `sum(f, (n, a, b))` and `product(f, (k, a, b))` are `\sum` and `\prod`. A name that runs a function into its argument, as `sinx`, is one name with a note in plain text, and refused in LaTeX and Unicode.
- A subscript on a symbol makes another symbol: `r_s` and `r_{s}` are both `r_s`. On a tensor, it is an index.
- A name followed by a bracket, with no space, is a call: `f(x)`. A function nothing defines is unknown, as `A(r)` in a metric. A function written as a command takes what follows it: `\sin x`, `\sin^2\theta`.
- Arithmetic is exact: `1/2` is a half, and a decimal stays a decimal. `^` and `**` are powers, a space between factors multiplies, and `299\,792\,458` is one number.
- `[1, 2]` is a list, which is a vector in arithmetic; a list of rows, or a matrix environment, is a matrix.
- A bar with values, as in `x^2 |_{x=2}`, puts the values in.

| Constant | Value |
|---|---|
| `pi` | π |
| `e` | Euler's number |
| `i` | the imaginary unit |
| `oo` | infinity |
| `infty` | infinity, as `\infty` |
| `True` | true |
| `False` | false |

## Index notation

- `^{...}` holds upper indices and `_{...}` lower ones. In braces each letter is an index and a Greek name is one: `_{ab}` is a, b and `_{\mu\nu}` (or `_{mu nu}`) is μ, ν. A script without braces is one letter, command or number, as in TeX: `R^a_b` is `R^{a}_{b}`. `{}` staggers indices: `R^{a}{}_{bcd}`.
- An index once up and once down in a term is summed; the others are free, and are the same in every term and on both sides. These rules are checked before anything is computed.
- With a metric in force, it raises and lowers indices: write a tensor with its indices where you want them.
- A component is chosen by number, `g_{01}`, by a coordinate's name, `\Gamma^{r}_{\theta\theta}`, or with `=` or `:`, as in `A^{\mu=0}` and `w^{a:MSFT}`.
- `\partial_\mu`, or `d_{a}`, and `\nabla_\mu`, or `D_{a}`, are the partial and the covariant derivative of the factor after them; `T^{\mu\nu}{}_{,\nu}` and `T^{\mu\nu}{}_{;\nu}` mean the same.
- `T_{(ab)}` symmetrises and `F_{[ab]}` antisymmetrises, also across tensors.
- An equation between tensors with components stands for one equation for each component. One between tensors without components, as `T_{\mu\nu}` before it is defined, is kept as an equation of notation.

### Built-in tensors

With a metric in force, these names mean the tensors of its geometry, by name and number of indices. The conventions are those of Misner, Thorne and Wheeler, and of Carroll.

| Name | Indices | Meaning |
|---|---|---|
| `g` | 2 | the metric, `g_{\mu\nu}`, and its inverse, `g^{\mu\nu}` |
| `C` `\Gamma` | 3 | the Christoffel symbols, `\Gamma^a{}_{bc}` |
| `R` | 4 | the Riemann tensor, `R^a{}_{bcd}` |
| `Ric` `R` | 2 | the Ricci tensor, `R_{ab}` |
| `G` | 2 | the Einstein tensor, `G_{ab}` |
| `W` `C` | 4 | the Weyl tensor, `C_{abcd}` |
| `\delta` | 2 | the Kronecker delta, `\delta^a{}_b` |
| `g` | 0 | the determinant of the metric |
| `Ric` `R` | 0 | the Ricci scalar |
| `Kretschmann` | 0 | the Kretschmann scalar, `R_{abcd} R^{abcd}` |
| `geodesic_equations(\tau)` |  | the geodesic equations, in the parameter `\tau` |

`R` and `g` stay symbols where the metric is written with them, as the radius R of a sphere is; `Ric` is always the Ricci scalar. In a line element, `d\Omega^2` is `d\theta^2 + \sin^2\theta\, d\phi^2`.

### Equations between tensors without components

| Operation | Meaning |
|---|---|
| `LHS` | its left side |
| `RHS` | its right side |
| `simplify` | its canonical form, with the identities of General Relativity |
| `solve` | `solve(E, T_{\mu\nu})`, for a tensor, with its indices up or down |
| `subs` | `subs(E, rule)`: a definition, a contraction, or a value such as `\Lambda = 0` |
| `trace` | contracts its two free indices with the metric |

## Vectors

- A list, a tuple or a column of numbers is a vector, as `\mathbf{v} := (1, 2, 3)`. Between vectors `\cdot` is the dot product and `\times` the cross product; between numbers both multiply. `|\mathbf{v}|` is its length, and `\mathbf{p}^2` is `\mathbf{p} \cdot \mathbf{p}`.
- A vector's derivatives, integrals, evaluation bars and limits are each component's; `\mathbf{v}(\mathbf{r}(t))` is a field at a point.
- `\hat{\mathbf{x}}`, `\hat{x}`, `\hat{\imath}` and `\mathbf{i}`, `\mathbf{j}`, `\mathbf{k}`, which nothing defines, are Cartesian unit vectors; `\hat{\mathbf{r}}` and `\mathbf{e}_r` are those of the coordinates in force.
- `\nabla f`, `\nabla \cdot \mathbf{v}`, `\nabla \times \mathbf{v}`, `\nabla^2 f` and `\Delta f` are of the coordinates in force, whose scale factors a line element gives, or the Cartesian coordinates written in them, or their names; without coordinates they are of x, y and z. `(\mathbf{u} \cdot \nabla)` acts on what follows it. Of a function at a point, `\nabla f(\mathbf{a})` is its gradient there and `\nabla^2 f(\mathbf{a})` its Hessian.
- An equation between vectors holds component by component; `\mathbf{0}` is the zero vector. `\iint_{x^2 + y^2 \le 1} f\,dA` is over the disk, `\iiint_{[0, 1]^3} f\,dV` over the cube, and `\oint_C \mathbf{F} \cdot d\mathbf{r}` along a curve `C(t) := ...` whose parameter's range is stated.
- Where the metric is written as the identity, `ds^2 = dx^2 + dy^2 + dz^2`, or a space is `\mathbb{R}^n`, an index repeated in any position is summed, and `\epsilon_{ijk}` is the Levi-Civita symbol.

## Spacetimes

Each is a function that gives a metric, as in `g_{\mu\nu} := Schwarzschild(M)`. Parameters are names or numbers, and units are geometric, G = c = 1, unless `G` and `c` follow the spacetime's own parameters: `Minkowski(coordinates, G, c)`, `Schwarzschild(M, G, c)`, `ReissnerNordstrom(M, Q, G, c)`, `Kerr(M, a, G, c)`, `KerrNewman(M, a, Q, G, c)`, `FLRW(k, a, G, c)`, `DeSitter(L, G, c)`, `AntiDeSitter(L, G, c)`.

## Functions

A function is called by its name, `simplify(x)`, or, if LaTeX has it, as a command, `\sin x`, or as books write it, `\Gamma(x)`: a definition of the name in the program comes first. Most apply to each component of a tensor: `simplify(R_{abcd})` simplifies every component. Each is a capability of an integration, a module of `relativisticpy.integrations` ([Integrations](integrations.md)), listed here by family, with every name and the forms books write it in; `relativisticpy help NAME` describes one in full.

- **Activation functions of neural networks:** `ReLU` `\operatorname{ReLU}(x)`, `softplus` `\operatorname{softplus}(x)`
- **Algebra and calculus:** `simplify` `tsimplify`, `expand`, `factor`, `collect`, `cancel`, `apart`, `together`, `trigsimp`, `powsimp`, `expand_trig`, `nsimplify`, `diff`, `integrate`, `limit` `lim`, `series`, `subs`, `N` `numerical` `evalf`, `doit`
- **Bessel and Airy functions:** `besselj` `J_{\nu}(x)`, `bessely` `Y_{\nu}(x)`, `besseli` `I_{\nu}(x)`, `besselk` `K_{\nu}(x)`, `jn` `j_n(x)`, `yn` `y_n(x)`, `airyai` `\operatorname{Ai}(x)`, `airybi` `\operatorname{Bi}(x)`
- **The notation of chemistry:** `colog` `\mathrm{p}X` `\mathrm{p}(X)`
- **Combinatorics:** `stirling2`, `stirling1`, `catalan`, `fibonacci`, `bell`, `bernoulli` `B_n`, `harmonic` `H_n`, `partition` `p(n)`, `multinomial`
- **Elementary functions:** `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin` `arcsin`, `acos` `arccos`, `atan` `arctan`, `acot`, `asec`, `acsc`, `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`, `asinh` `arsinh`, `acosh` `arcosh`, `atanh` `artanh`, `acoth` `arcoth`, `asech` `arsech`, `acsch` `arcsch`, `exp`, `ln`, `sqrt`, `abs` `Abs`, `sign`, `floor`, `ceiling`, `factorial`, `factorial2`, `gamma` `\Gamma(x)`, `re` `Re`, `im` `Im`, `conjugate` `conj`, `arg`, `erf`
- **Equations:** `solve`, `dsolve`, `RHS` `rhs`, `LHS` `lhs`, `Eq`, `symbols`, `latex`
- **The discrete Fourier transform:** `fft` `\operatorname{fft}(x)`, `ifft` `\operatorname{ifft}(X)`
- **Matrices:** `diag`, `det`, `inverse` `inv`, `pinv`, `transpose`, `ctranspose`, `trace` `tr` `Tr` `\operatorname{tr} A` `\operatorname{tr}(A)` `\operatorname{Tr} A` `\operatorname{Tr}(A)`, `rank` `\operatorname{rank} A` `\operatorname{rank}(A)`, `adjugate` `adj` `\operatorname{adj} A` `\operatorname{adj}(A)`, `null` `\operatorname{null} A` `\operatorname{null}(A)`, `dim`, `vec` `\operatorname{vec} A` `\operatorname{vec}(A)`, `norm`, `cond` `\kappa(A)` `\kappa_p(A)`, `spectral_radius` `\rho(A)`, `lambda_max` `\lambda_{\max}(A)` `\lambda_{\max} A`, `lambda_min` `\lambda_{\min}(A)` `\lambda_{\min} A`, `eigenvalue` `\lambda_k(A)`, `spec` `\operatorname{spec}(A)` `\sigma(A)`, `expm`, `logm`, `sqrtm`, `matrix_power`, `kron`, `dot`, `cross`, `direct_sum`
- **Number theory:** `mod` `Mod`, `totient` `\varphi(n)` `\phi(n)`, `mobius` `\mu(n)`, `primepi` `\pi(x)`, `divisor_sigma` `\sigma_k(n)`, `divisor_count` `d(n)`, `primenu` `\omega(n)`, `primeomega` `\Omega(n)`, `mangoldt` `\Lambda(n)`
- **Numbers:** `log`, `root`, `binomial`, `atan2`, `min` `Min`, `max` `Max`, `gcd`, `lcm`
- **Orthogonal polynomials and spherical harmonics:** `legendre` `P_n(x)`, `assoc_legendre` `P_{\ell}^m(x)`, `hermite` `H_n(x)`, `laguerre` `L_n(x)`, `assoc_laguerre` `L_n^{(\alpha)}(x)`, `chebyshevt` `T_n(x)`, `chebyshevu` `U_n(x)`, `Ynm` `Y_{\ell}^m(\theta, \phi)`
- **Sequences, as signal processing writes them:** `impulse` `\delta[n]`, `step` `u[n]`, `poles`, `zeros`
- **Special functions:** `beta` `\mathrm{B}(a, b)`, `zeta` `\zeta(s)`, `Log`, `erfc`, `erfi`, `Ei`, `Si`, `Ci`, `LambertW` `W_k(x)`, `elliptic_k` `\operatorname{K}(m)`, `elliptic_e` `\operatorname{E}(m)`, `hyper` `{}_pF_q(a; b; z)`, `rf` `RisingFactorial` `(a)_n`, `normcdf` `\Phi(x)`, `Heaviside` `u_c(t)`, `DiracDelta` `\operatorname{DiracDelta}(x)` `\delta(x)`
- **Integral transforms:** `LaplaceTransform` `laplace_transform` `\mathcal{L}\{f\}(s)` `\mathcal{L}\{f\}`, `InverseLaplaceTransform` `inverse_laplace_transform` `\mathcal{L}^{-1}\{F\}(t)` `\mathcal{L}^{-1}\{F\}`, `FourierTransform` `fourier_transform` `\mathcal{F}\{f\}(s)` `\mathcal{F}\{f\}`, `InverseFourierTransform` `inverse_fourier_transform` `\mathcal{F}^{-1}\{F\}(x)` `\mathcal{F}^{-1}\{F\}`, `ZTransform` `z_transform` `\mathcal{Z}\{x\}(z)` `\mathcal{Z}\{x\}`, `InverseZTransform` `inverse_z_transform` `\mathcal{Z}^{-1}\{X\}(n)` `\mathcal{Z}^{-1}\{X\}`
- **Rewrites,** what a program writes, computed faster by another capability, to the same numbers: `dft_sum` `\sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}`, by `fft`

### Optimizers

An optimum, `\min_{x} f(x)`, `\arg\max_{x, y} u \quad \text{subject to} \quad g = 0` or `\sup_{0 \le x < 1} f(x)`, is found by the optimizers, in this order: those that compute exactly, `extrema`, `kkt`, `simplex`, `enumeration`; then, where it has no parameters, those that compute numerically, `linprog`, `quadratic`, `minimize`, which take first one written with decimals, as data is. A note says which found it, how, and whether it is global. Over an index of data, `\max_t x_t` is the greatest value along it, and `\arg\max_t x_t` its label.

### Book names not supported yet

A name books write for a function no integration provides is refused, saying what it is, never read as an unknown function or a symbol: `\operatorname{Res}_{z=a} f`, the residue of a function at a point; `\epsilon_{ijk}`, the Levi-Civita symbol.

## Data, statistics and linear algebra

Data is bound from Python, `wb["R_{ta}"] = array`, and computed with NumPy: `pip install "relativisticpy[numeric]"`, or `[data]` for pandas and scikit-learn. An index ranges over a space: of coordinates, assets, days or states. `1_a` is the covector of ones, `A_{ta} \odot B_{ta}` the elementwise product and `\|v\|` the length of a vector.

### Statistics

A statistic binds its index, as a sum does: inside it, every factor shares t.

| Statistic | Meaning |
|---|---|
| `\mathbb{E}_t[X]` | the expectation over t, a contraction with its measure; named `𝔼` |
| `\operatorname{Var}_t(X)` | the variance over t of each component; named `Var`, `var` |
| `\operatorname{Cov}_t(X, Y)` | the covariance over t, with the free indices of both; named `Cov`, `cov` |
| `\operatorname{LedoitWolf}_t(R_{ta})` | the Ledoit–Wolf estimate of a covariance, with scikit-learn; named `LedoitWolf` |

### Linear algebra

Given a tensor, these functions are linear algebra. They are written `\operatorname{eig}`, or as a command where LaTeX has one, `\det`.

| Function | Meaning |
|---|---|
| `det` | the determinant of a map |
| `diag` | `\operatorname{diag}(d)_{ab}`, the diagonal matrix of a vector |
| `eig` | `\lambda_k, v^a{}_k := \operatorname{eig}(A^a{}_b)`: eigenvalues and eigenvectors; `eig(A)` of a matrix written without indices, its eigenvalues, as MATLAB gives them |
| `exp` | the exponential of a tensor named without indices: `\exp(-i H t)` |
| `inv` | the inverse; `(\Sigma^{-1})^{ab}` is written as on paper, and solved for, not formed |
| `log` | the logarithm of a matrix |
| `spec` | `\operatorname{spec}(A^a{}_b)`, the eigenvalues; with a second argument n, the n largest, or smallest if n is negative |
| `sqrt` | the square root of a matrix, `\sqrt{\Sigma}` |
| `svd` | `U_{tk}, s_k, V_{ak} := \operatorname{svd}(X_{ta})`: the singular values and vectors |
| `tr` `trace` | the trace of a map |

### Python objects

`\texttt{prices}` is the object called prices in the Python scope a workbook is given, `Workbook(scope=globals())`, or of the objects passed to `show`. Attributes, keys and positions follow, `\texttt{prices["NVDA"]}`, and are read, never run as Python. A document checked, and a program sent to the MCP server, have no scope. Defined with indices, `P_{ta} := \texttt{prices}`, data is bound as `wb["P_{ta}"] = prices` binds it, and meets other data by label; written with them, `\texttt{signal}_t`, it has them. What an object becomes, by the object integrations:

| Object | Becomes |
|---|---|
| `numpy.ndarray` `numpy.generic` `anything with __array__` `anything with __array_namespace__` `anything with __dlpack__` | an array, or anything NumPy reads: data, each axis over a space of its size; a NumPy number is a scalar |
| `pandas.DataFrame` `pandas.Series` | a DataFrame or Series: labelled data, its index and columns the bases of spaces named after its axes |
| `int` `float` `complex` `bool` `fractions.Fraction` `decimal.Decimal` `numbers.Number` | a number: a scalar, exact where Python's is, as an integer or a fraction |
| `list` `tuple` | a list or tuple of numbers, or of lists of them: a vector or a matrix, as the language's own lists are |
| `anything callable` | a function, or anything called: called in a program, on numbers, and on data elementwise |
| `scipy.sparse.spmatrix` `scipy.sparse.sparray` | a SciPy sparse matrix or array: sparse data, never made dense |
| `sympy.Basic` `sympy.matrices.MatrixBase` `sympy.NDimArray` `anything with _sympy_` | a SymPy expression, or anything with _sympy_: an expression, its symbols the program's by name; a SymPy matrix is a matrix |
| `xarray.DataArray` | a DataArray: data over its named dimensions, labelled by their coordinates |

## Random variables

`X \sim D` declares a random variable, `X, Y \overset{\text{iid}}{\sim} D` independent ones, `X_1, \ldots, X_n \overset{\text{iid}}{\sim} D` a sample, and `X \mid N \sim D` one given another. A distribution is written as books write it, or called by its name with its parameters as SymPy takes them: `Normal(mu, sigma)` `\mathcal{N}(\mu, \sigma^2)` `N(\mu, \sigma^2)`, `Poisson(rate)` `\operatorname{Poisson}(\lambda)` `\operatorname{Pois}(\lambda)`, `Binomial(n, p)` `\operatorname{Bin}(n, p)` `\operatorname{Binom}(n, p)`, `Bernoulli(p)` `\operatorname{Bernoulli}(p)` `\operatorname{Ber}(p)` `\operatorname{Bern}(p)`, `Uniform(a, b)` `U(a, b)` `\mathcal{U}(a, b)` `\operatorname{Unif}(a, b)`, `Exponential(rate)` `\operatorname{Exp}(\lambda)`, `GammaDistribution(k, theta)` `\Gamma(k, \theta)`, `ChiSquared(k)` `\chi^2_k` `\chi_k^2` `\chi^2(k)`, `StudentT(nu)` `t_\nu` `t(\nu)`, `BetaDistribution(alpha, beta)` `\operatorname{Beta}(\alpha, \beta)`, `Geometric(p)` `\operatorname{Geom}(p)`.

`\Pr(X > 1)`, `\mathbb{P}(A)`, `P(X = 2)`, `\mathbb{E}[X]`, `\operatorname{Var}(X)` and `\operatorname{Cov}(X, Y)` are computed, with `\mid` the conditional bar; `f_X(x)`, `F_X(x)`, `F_X^{-1}(q)`, `M_X(t)` and `\varphi_X(t)` are X's density, distribution function, quantile and generating functions: exactly by SymPy, or by SciPy where SymPy finds no closed form in 5 s and there are no parameters. `\Pr(A) := 0.3` gives an event's probability, and a capital letter nothing defines is a random variable of any law.

## LaTeX

Typesetting is ignored: spacing, sizes, `\label`, `\tag`, `$`, `$$`, `\[ \]` and the `&` of alignments. Every statement inside a display is LaTeX, however it is written. An unknown command is an error that suggests the closest one.

### Commands

| Command | Meaning |
|---|---|
| `\alpha \beta \chi \Delta \delta \epsilon \eta \Gamma \gamma \iota \kappa \Lambda \lambda \mu \nu \Omega \omega \omicron \Phi \phi \Pi \pi \Psi \psi \rho \Sigma \sigma \tau \Theta \theta \Upsilon \upsilon \Xi \xi \zeta` | Greek letters, each one symbol |
| `\varDelta \varepsilon \varGamma \varkappa \varLambda \varOmega \varPhi \varphi \varPi \varPsi \varrho \varSigma \varsigma \varTheta \vartheta \varUpsilon \varXi` | variant letterforms: each the letter after `var` |
| `\dagger \ell \hbar \imath \infty \intercal \jmath \odot \oplus \star \top \varpi` | symbols; `\odot` is the Sun, as in `M_\odot`, and between tensors the elementwise product; `\oplus` between matrices is their direct sum, `A^\top`, `A^\intercal` and `A^\dagger` of a matrix its transpose and its conjugate transpose, and `\imath` and `\jmath` are i and j, under a hat unit vectors |
| `\arccos \arccot \arcsin \arctan \arg \cos \cosh \cot \coth \csc \csch \det \dim \exp \gcd \Im \ker \ln \log \max \min \Pr \Re \sec \sech \sgn \sin \sinh \tan \tanh` | functions, each calling the builtin of its name, except that `\arccos` calls `acos`, `\arccot` calls `acot`, `\arcsin` calls `asin`, `\arctan` calls `atan`, `\Im` calls `im`, `\ker` calls `null`, `\Re` calls `re`, `\sgn` calls `sign`; `\sin^{-1}` is the inverse, and only `\log` takes a subscript, its base; `\Pr` is a probability |
| `\bar \breve \check \ddot \dot \hat \overline \tilde \vec \widehat \widetilde` | accents: `\dot{a}` and `\ddot{a}` are time derivatives of a function of one variable, `\hat{x}` on a coordinate a unit vector (see Vectors); the others name a new symbol |
| `\ast \cdot \otimes \times` | multiplication, `*`; between tensors, their product, `\otimes` between matrices their Kronecker product, and `\cdot` and `\times` between vectors their dot and cross products |
| `\div` | division, `/` |
| `\lt \gt \le \leq \leqslant \ge \geq \geqslant \ne \neq` | comparisons, in this order: `<`, `>`, `<=` (three spellings), `>=` (three), `!=` (two) |
| `\longrightarrow \mapsto \rightarrow \to` | an arrow, as in `\lim_{x \to 0}` |
| `\coloneq \coloneqq \defeq \equiv \triangleq` | a definition, `:=` |
| `\Vert \lVert \lvert \mid \rVert \rvert \vert` | a bar: absolute values, norms, evaluation bars, and the bar of a bra or a ket |
| `\langle \rangle` | angle brackets: Dirac's notation, with the bar of a bra or a ket; without a bar, an expectation value, `\langle x \rangle`, or an inner product, `\langle u, v \rangle` |
| `\pm \mp` | both signs, so two values |
| `\wedge` | the wedge product |
| `\circ` | composition: read, not computed |
| `\approx \asymp \cong \doteq \gg \in \leftrightarrow \ll \longleftrightarrow \notin \prec \preceq \propto \sim \simeq \subset \subseteq \succ \succeq \supset \supseteq` | relations, read so that the equations around them run, but not computed; `\in` also declares index labels |
| `\iff \implies \Leftrightarrow \Longleftrightarrow \Longrightarrow \Rightarrow` | between equations, each side a statement |
| `\cdots \ddots \dots \dotsb \dotsc \dotsi \dotsm \dotso \ldots \vdots` | terms not written: read, not computed |
| `\hbox \mbox \operatorname \text \textbf \textit \textnormal \textrm \textsf \textup` | text: one name, as `\text{otherwise}`; `\operatorname{name}` names a function |
| `\texttt` | a Python object, read by name from the workbook's scope: `\texttt{prices}` (see Python objects) |
| `\bm \boldsymbol \lefteqn \mathbb \mathbf \mathbin \mathcal \mathclose \mathfrak \mathinner \mathit \mathnormal \mathop \mathopen \mathord \mathpunct \mathrel \mathrm \mathscr \mathsf \mathtt \pmb \substack` | typefaces: what they hold is read as it is, so `\mathrm{d}x` is `d x`; bold, calligraphic, script, Fraktur and sans serif make a name of its own where the program defines it in two |
| `\bf \bfseries \boldmath \cal \em \it \itshape \mdseries \mit \normalfont \rm \rmfamily \sc \sf \sl \tt \upshape` | plain TeX's typefaces, for the rest of the group: `{\rm d}x` |
| `\aftergroup \allowbreak \centering \displaybreak \displaystyle \enspace \hfill \hline \hphantom \hspace \label \limits \medspace \negmedspace \negthickspace \negthinspace \noindent \nolimits \nonumber \notag \phantom \qquad \quad \scriptscriptstyle \scriptstyle \tag \textstyle \thickspace \thinspace \vphantom \vspace` | layout, ignored; `\quad` and `\qquad` also separate equations |
| `\Big \Bigg \Biggl \Biggm \Biggr \Bigl \Bigm \Bigr \big \bigg \biggl \biggm \biggr \bigl \bigm \bigr` | delimiter sizes, ignored |
| `\linebreak \newline` | end a statement, as `\\` does |
| `\cases \matrix \pmatrix` | plain TeX's matrices and cases: `\matrix{a & b \cr c & d}` |
| `\cfrac \dfrac \frac \over \tfrac` | fractions, `\frac{a}{b}` and plain TeX's `{a \over b}`; `\frac{d}{dx}` and `\frac{\partial f}{\partial x}` are derivatives |
| `\sqrt` | roots, `\sqrt{x}` and `\sqrt[n]{x}` |
| `\binom \choose \dbinom \tbinom` | binomial coefficients, `\binom{n}{k}` and `{n \choose k}` |
| `\brace \brack` | the Stirling numbers in Karamata's brackets, `{n \brace k}` and `{n \brack k}` |
| `\bmod \mod \pmod` | the remainder, `a \bmod m`, and a congruence, `a \equiv b \pmod{m}` |
| `\circledast` | the circular convolution of sequences, `(x \circledast h)[n]`, of length N with `\circledast_N` |
| `\prod \sum` | a sum, `\sum_{n=1}^{N}`, and a product, in closed form where SymPy finds one |
| `\iiint \iint \int \oint` | integrals, single, double and triple, which end at their differentials: `\int_0^1 f \, dx` |
| `\lim` | a limit, `\lim_{x \to 0}`, or from one side, `\lim_{x \to 0^+}` |
| `\inf \sup \underset` | the supremum and the infimum over a variable, `\sup_{0 \le x < 1} x^n`, as `\max_x` and `\arg\min_w` are optima; `\underset{x}{\min}` sets the variable beneath |
| `\nabla \partial` | the partial and the covariant derivative, `\partial_\mu` and `\nabla_\mu`; `\partial_t^2` is the second derivative, and `\nabla` alone vector calculus's |
| `\Box \square` | the d'Alembertian, `\nabla^\mu \nabla_\mu`, as is `\nabla^2` with a metric in force; without one, `\nabla^2` is the Laplacian |
| `\pounds` | the Lie derivative, `\pounds_V T_{ab}` |
| `\dv \pdv` | the physics package's derivatives: `\pdv{f}{x}`, `\pdv[2]{f}{x}`, `\dv{f}{x}` |
| `\bra \braket \expval \ket \ketbra \mel` | Dirac's notation as the physics and braket packages write it: `\ket{\psi}`, `\bra{\phi}`, `\braket{\phi}{\psi}`, `\ketbra{a}{b}`, `\expval{A}` and `\mel{\phi}{A}{\psi}` |
| `\lceil \lfloor \rceil \rfloor` | the floor `\lfloor x \rfloor` and the ceiling |
| `\left \middle \right` | `\left( ... \right)` groups, with any delimiters; between bars it is an absolute value, from `\left.` to a bar with `_{x=a}` an evaluation bar, and between angle brackets, with `\middle` before the bar, Dirac's notation |
| `\begin \end` | an environment, `\begin{pmatrix} ... \end{pmatrix}` |
| `\bgroup \egroup` | plain TeX's braces |
| `\cr` | ends a row of plain TeX's `\matrix` and `\cases` |
| `\overset \stackrel` | a label over a relation: `X_1, \ldots, X_n \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)` declares independent random variables |
| `\mathbb` | blackboard bold: `\mathbb{R}^{n}` and `\mathbb{C}^{n}` are spaces, `\mathbb{E}_t` an expectation and `\mathbb{1}_a` the covector of ones |

### Environments

| Environment | Meaning |
|---|---|
| `array`, `Bmatrix`, `bmatrix`, `matrix`, `pmatrix`, `smallmatrix`, `Vmatrix`, `vmatrix` | a matrix; `vmatrix` is its determinant |
| `cases`, `dcases` | a piecewise function, with `\text{if}` and `\text{otherwise}` |
| `align`, `align*`, `alignat`, `alignat*`, `aligned`, `alignedat`, `dgroup`, `dgroup*`, `dmath`, `dmath*`, `eqnarray`, `eqnarray*`, `equation`, `equation*`, `flalign`, `flalign*`, `gather`, `gather*`, `gathered`, `IEEEeqnarray`, `IEEEeqnarray*`, `math`, `multline`, `multline*`, `split` | statements, separated by `\\` |

- In `align`, `align*`, `alignat`, `alignat*`, `aligned`, `alignedat`, `flalign`, `flalign*`, each pair of columns is an equation; `multline`, `multline*` holds one equation, broken over rows.
- `alignat`, `alignat*`, `alignedat`, `array`, `IEEEeqnarray`, `IEEEeqnarray*` take an argument first, such as the columns of `array`.
- `relativisticpy check` reads the equation environments of a document as displays, with `displaymath`, `$$ ... $$` and `\[ ... \]`; it reads `aligned`, `alignedat`, `gathered`, `split` inside them.

### Not supported

These commands are LaTeX's, or a package's, but not the language's. Each is an error that says what to write instead:

- `\abs`: the physics package's absolute value; write `|x|`, or `\left| x \right|`.
- `\dd`: the physics package's differential; write `d x`, or `\mathrm{d} x`.
- `\def \let \newcommand \providecommand \renewcommand`: defines a macro, which a program cannot; write what it stands for; relativisticpy check reads the macros of a document.
- `\eval`: the physics package's evaluation bar; write `\left. f \right|_{x=0}`.
- `\exists`: the quantifier; write `\text{there is}`.
- `\forall`: the quantifier; write `\text{for all}`.
- `\liminf \limsup`: write `\lim_{x \to a}` for a limit.
- `\norm`: the physics package's norm; write `\| x \|`.
- `\not`: write `\neq` for not equal, or `\notin`.
- `\order`: the physics package's order; write the terms out, or `\mathcal{O}(x^2)` as a symbol.
- `\qty`: the physics package's brackets; write `\left( ... \right)`.
- `\Tr`: the physics package's trace; write `\operatorname{Tr}(A)`, or contract the indices: `A^{a}_{a}`.
- `\tr`: the physics package's trace; write `\operatorname{tr}(A)`, or contract the indices: `A^{a}_{a}`.
- `\va`: the physics package's vector arrow; write `\vec{v}`.
- `\vb`: the physics package's bold vector; write `\mathbf{v}`.

### Unicode

Unicode reads as the LaTeX it stands for. A Greek letter, `θ`, is its command, and so are ∂ `\partial`, ∇ `\nabla`, √ `\sqrt`, ∑ `\sum`, ∏ `\prod`, ∫ `\int`, ∬ `\iint`, ∭ `\iiint`, ∮ `\oint`, ⊛ `\circledast`, ⌊ `\lfloor`, ⌋ `\rfloor`, ⌈ `\lceil`, ⌉ `\rceil`. The operators · ⋅ ∗ × ÷ ⁄ − ≤ ≥ ≠ → ↦ ≡ ≔ ′ ± ∓ ∧ ∘ ⊗ ‖ ⟨ ⟩ 〈 〉 … ⋯ and the relations ∼ ≃ ≈ ≅ ∝ ≪ ≫ ∈ ∉ ⊂ ⊆ ↔ are LaTeX's. Superscripts and subscripts are scripts, letters too, so `r²` is `r^{2}`, `rₛ` is `r_{s}` and `Aᵀ` is `A^{T}`, † after a matrix is `^\dagger`, ⊕ is `\oplus`, and the double-struck letters, ℝ ℂ 𝔼 𝟙, are `\mathbb`'s. ∛ and ∜ are `\sqrt[3]` and `\sqrt[4]`, and a vulgar fraction, ½, and the fraction slash, `3⁄4`, are fractions. A combining accent, as in ȧ, is an error that names it: write the command, `\dot{a}`.

## Verdicts of a check

`relativisticpy check paper.tex`, or `wb.check(document)` in Python, runs the display equations of a document in order and gives each statement a verdict.

| Mark | Verdict | Meaning | Fails `--strict` |
|---|---|---|---|
| ✓ | holds | proven equal | no |
| ✗ | differs | of known quantities, whose sides are not equal: a mistake, or a condition | yes |
| ↔ | relates | relations between quantities nothing defines, not identities | no |
| ✗ | ill-formed | notation that breaks the rules of index notation | yes |
| = | defines | definitions | no |
| → | computes | expressions, computed | no |
| ~ | not computed | relations read but not decided, or over the time limit | over a time limit |
| ? | not understood | what RelativisticPy cannot do yet | no |
| ! | not parsed | not read | yes |

`--strict` also fails on a document in which no equation was found. A report lists the statements that are `differs` `ill-formed` `not parsed`, and counts the rest; `--all` lists every one. In Python, `report.failed(...)` takes the verdicts to fail on, and `time limit` and `empty` besides.

## Errors

Every error a user can cause is a `RelativisticPyError`. It says what is wrong, where, as `--> file:line:column` with the text underlined, why, in a `note:`, and how to fix it, in a `hint:`. `error.to_dict()` gives the same as data.

| Error | When |
|---|---|
| `ParseError` | the source text is not valid RelativisticPy (raised by the lexer and parser) |
| `CompileError` | a well-formed statement does not make sense: an unknown tensor, a function called with the wrong number of arguments, a metric without coordinates |
| `IndexNotationError` | tensor indices break the rules of index notation: mismatched free indices in a sum, an index repeated three times, two lower indices summed together |
| `NotComputedError` | a statement RelativisticPy reads but does not compute: a relation such as `≈`, `∝` or `→`, a condition on a coordinate, `\theta = \pi/2`, or a relation between differentials |
| `EvaluationError` | a computation failed while running: a degenerate metric, an equation SymPy cannot solve |
| `MetricError` | a metric that cannot be one: not square, not symmetric, or degenerate |
| `DataError` | data bound from Python, as by `wb["w^a"] = ...`, cannot be bound as written: it is ragged, it does not have an axis for each index, or its labels are not the elements of its space |
| `UnknownNameError` | a name looked up from Python, as in `wb["T"]`, that nothing defines |
| `UsageError` | a call from Python was given what it cannot take: `wb.exe(123)`, a backend that does not exist, a time limit of no seconds |
| `FileError` | a file cannot be read: it does not exist, or it is not UTF-8 text |
| `UndecidedError` | an equation was asked for a truth value it does not have, or a program claims one that is not decided: SymPy could not show its sides equal, nor numbers show that they differ |
| `InternalError` | a failure of RelativisticPy's own: never a mistake in the input |
