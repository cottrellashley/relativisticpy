# The language reference

> **For:** anyone, person or agent, who needs to know what the language accepts. **You will learn:** every statement, command, environment, function, constant, spacetime, statistic, verdict and error, on one page.

This page is generated from the tables the language is read with. `relativisticpy help reference` prints it offline, and `relativisticpy.reference.markdown()` gives it in Python. The guides explain each part, with examples: [The language](language.md), [LaTeX](latex.md), [Tensors and geometry](tensors.md) and [Beyond spacetime](spaces.md).

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
| `Coordinates := [t, r, \theta, \phi]` | the coordinates, in order |
| `Constants` `Reals` `Integers` `Complex` `Positive` `:= [a, b]` | declare a and b constants, real, integers, complex or positive, before they are used |

A renaming statement gives a built-in another name: `MetricSymbol := G` makes the metric `G`. They are, with the names they change: `MetricSymbol` (`g`), `ConnectionSymbol` (`C`), `RiemannSymbol` (`R`), `RicciSymbol` (`Ric`), `EinsteinSymbol` (`G`), `EinsteinTensorSymbol` (`G`), `WeylSymbol` (`W`), `DerivativeSymbol` (`d`), `CovariantDerivativeSymbol` (`D`).

## Names and numbers

- A name that nothing defines is a symbol. `theta`, `\theta` and `θ` are one symbol, as is every spelling of a Greek letter.
- In LaTeX, a run of letters is a product of one symbol for each letter: `\frac{2GM}{r}` is 2·G·M/r. Greek names, defined names and builtins stay whole, and `\mathrm{GM}` or `\text{GM}` is one name. In plain text, `GM` is one name.
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

## Spacetimes

Each is a function that gives a metric, as in `g_{\mu\nu} := Schwarzschild(M)`. Parameters are names or numbers, and units are geometric, G = c = 1, unless `G` and `c` follow the spacetime's own parameters.

| Spacetime | Coordinates | Meaning |
|---|---|---|
| `Minkowski(coordinates, G, c)` | t, x, y, z | flat spacetime; `Minkowski(spherical)` in spherical coordinates |
| `Schwarzschild(M, G, c)` | t, r, θ, φ | a static black hole of mass M |
| `ReissnerNordstrom(M, Q, G, c)` | t, r, θ, φ | a static black hole of mass M and charge Q |
| `Kerr(M, a, G, c)` | t, r, θ, φ | a rotating black hole of mass M and spin a = J/Mc, in Boyer–Lindquist coordinates |
| `KerrNewman(M, a, Q, G, c)` | t, r, θ, φ | a rotating black hole of mass M, spin a and charge Q |
| `FLRW(k, a, G, c)` | t, r, θ, φ | a homogeneous, isotropic universe: spatial curvature k, scale factor a(t) |
| `DeSitter(L, G, c)` | t, r, θ, φ | the static patch of de Sitter space, of curvature radius L |
| `AntiDeSitter(L, G, c)` | t, r, θ, φ | anti-de Sitter space, of curvature radius L |

## Functions

A function is called by its name, `simplify(x)`, or, if LaTeX has it, as a command, `\sin x`, or as books write it, `\Gamma(x)`: a definition of the name in the program comes first. Most apply to each component of a tensor: `simplify(R_{abcd})` simplifies every component. Each is a capability of an integration, a module of `relativisticpy.integrations` ([Integrations](integrations.md)), from whose declaration this list is written.

### Algebra and calculus

| Function | Arguments | Meaning |
|---|---|---|
| `simplify(expr)` `tsimplify` | 1 | the simplest form; of an equation between tensors without components, its canonical form |
| `expand(expr)` | 1 | multiplied out |
| `factor(expr)` | 1 | factorised |
| `collect(expr, syms, ...)` | 2 or more | `collect(f, x)`, the terms of f gathered by powers of x |
| `cancel(expr)` | 1 | a fraction, with common factors cancelled |
| `apart(expr)` | 1 | partial fractions |
| `together(expr)` | 1 | one fraction |
| `trigsimp(expr)` | 1 | trigonometric functions simplified |
| `powsimp(expr)` | 1 | powers combined: `x^a x^b` is `x^{a+b}` |
| `expand_trig(expr)` | 1 | functions of sums and multiples of angles, expanded |
| `nsimplify(expr)` | 1 | the exact number a decimal stands for: `0.5` is `1/2` |
| `diff(f, ...)` | 1 or more | `diff(f, x)`, `diff(f, x, 2)`, `diff(f, x, y)`: derivatives, as `\frac{d}{dx}` |
| `integrate(f, ...)` | 1 or more | `integrate(f, x)`, `integrate(f, (x, 0, 1))`: integrals, as `\int` |
| `limit(f, x, a, dir)` `lim` | 3 to 4 | `limit(f, x, a)`, as x tends to a; `\lim_{x \to a^+}` for one side |
| `series(expr, x, x0, n)` | 1 to 4 | `series(f, x, a, n)`, the Taylor series about a, to order n |
| `subs(expr, old, new)` | 2 to 3 | `subs(f, x, a)`, f with a put for x; `subs(f, x = a)` too |
| `N(value, digits)` `numerical` `evalf` | 1 to 2 | `N(x)`, `N(x, digits)`: a decimal number |
| `doit(value)` | 1 | sums, products and integrals, kept as written, carried out |

### Bessel and Airy functions

| Function | Arguments | Meaning |
|---|---|---|
| `besselj(nu, x)` `J_{\nu}(x)` | 2 | the Bessel function of the first kind, of order `\nu` |
| `bessely(nu, x)` `Y_{\nu}(x)` | 2 | the Bessel function of the second kind, of order `\nu` |
| `besseli(nu, x)` `I_{\nu}(x)` | 2 | the modified Bessel function of the first kind, of order `\nu` |
| `besselk(nu, x)` `K_{\nu}(x)` | 2 | the modified Bessel function of the second kind, of order `\nu` |
| `jn(n, x)` `j_n(x)` | 2 | the spherical Bessel function of the first kind, of order n |
| `yn(n, x)` `y_n(x)` | 2 | the spherical Bessel function of the second kind, of order n |
| `airyai(x)` `\operatorname{Ai}(x)` | 1 | the Airy function Ai |
| `airybi(x)` `\operatorname{Bi}(x)` | 1 | the Airy function Bi |

### Elementary functions

| Function | Arguments | Meaning |
|---|---|---|
| `sin(x)` | 1 | the sine; `\sin^2\theta` is the square of `\sin\theta` |
| `cos(x)` | 1 | the cosine |
| `tan(x)` | 1 | the tangent |
| `cot(x)` | 1 | the cotangent |
| `sec(x)` | 1 | the secant |
| `csc(x)` | 1 | the cosecant |
| `asin(x)` `arcsin` | 1 | the inverse sine; `\sin^{-1} x` too |
| `acos(x)` `arccos` | 1 | the inverse cosine |
| `atan(x)` `arctan` | 1 | the inverse tangent |
| `acot(x)` | 1 | the inverse cotangent |
| `asec(x)` | 1 | the inverse secant |
| `acsc(x)` | 1 | the inverse cosecant |
| `sinh(x)` | 1 | the hyperbolic sine |
| `cosh(x)` | 1 | the hyperbolic cosine |
| `tanh(x)` | 1 | the hyperbolic tangent |
| `coth(x)` | 1 | the hyperbolic cotangent |
| `sech(x)` | 1 | the hyperbolic secant |
| `csch(x)` | 1 | the hyperbolic cosecant |
| `asinh(x)` | 1 | the inverse hyperbolic sine |
| `acosh(x)` | 1 | the inverse hyperbolic cosine |
| `atanh(x)` | 1 | the inverse hyperbolic tangent |
| `acoth(x)` | 1 | the inverse hyperbolic cotangent |
| `asech(x)` | 1 | the inverse hyperbolic secant |
| `acsch(x)` | 1 | the inverse hyperbolic cosecant |
| `exp(x)` | 1 | the exponential; `e^{x}` too |
| `ln(x)` | 1 | the natural logarithm |
| `sqrt(x)` | 1 | the square root; `\sqrt[n]{x}` is the nth root |
| `abs(x)` `Abs` | 1 | the absolute value; `\lvert x \rvert` too |
| `sign(x)` | 1 | the sign: −1, 0 or 1 |
| `floor(x)` | 1 | rounded down; `\lfloor x \rfloor` too |
| `ceiling(x)` | 1 | rounded up; `\lceil x \rceil` too |
| `factorial(x)` | 1 | the factorial; `n!` too |
| `factorial2(x)` | 1 | the double factorial; `n!!` too |
| `gamma(x)` `\Gamma(x)` | 1 | Euler's gamma function; `\gamma` is the letter |
| `re(x)` `Re` | 1 | the real part; `\Re` too |
| `im(x)` `Im` | 1 | the imaginary part; `\Im` too |
| `conjugate(x)` `conj` | 1 | the complex conjugate |
| `arg(x)` | 1 | the argument of a complex number |
| `erf(x)` | 1 | the error function |

### Equations

| Function | Arguments | Meaning |
|---|---|---|
| `solve(equations, ...)` | 1 or more | `solve(x^2 = 4, x)`, the solutions; `solve(E, T_{\mu\nu})` solves an equation between tensors for one |
| `dsolve(equations, ...)` | 1 or more | `dsolve(f'(x) = f(x), f(x))`, the solution of a differential equation |
| `RHS(equation)` `rhs` | 1 | the right side of an equation |
| `LHS(equation)` `lhs` | 1 | the left side of an equation |
| `Eq(lhs, rhs)` | 2 | `Eq(a, b)`, the equation a = b |
| `symbols(names)` | 1 | `symbols('x y')`, the symbols named in quotes |
| `latex(value)` | 1 | the LaTeX of a value, as text |

### Matrices

| Function | Arguments | Meaning |
|---|---|---|
| `diag(first, ...)` | 1 or more | `diag(a, b, ...)`, a diagonal matrix |
| `det(matrix)` | 1 | the determinant |
| `inverse(matrix)` `inv` | 1 | the inverse of a matrix |
| `transpose(matrix)` | 1 | the transpose of a matrix |
| `trace(matrix)` | 1 | the trace of a matrix; of an equation between tensors, its contraction |

### Numbers

| Function | Arguments | Meaning |
|---|---|---|
| `log(x, base)` | 1 to 2 | the logarithm; `log(x, b)` to the base b, `\log_b x` in LaTeX |
| `root(x, n)` | 2 | `root(x, n)`, the nth root |
| `binomial(n, k)` | 2 | `binomial(n, k)`; `\binom{n}{k}` too |
| `atan2(y, x)` | 2 | `atan2(y, x)`, the angle of the point (x, y) |
| `min(first, ...)` `Min` | 1 or more | the least of the arguments |
| `max(first, ...)` `Max` | 1 or more | the greatest of the arguments |
| `gcd(a, b, ...)` | 2 or more | the greatest common divisor |
| `lcm(a, b, ...)` | 2 or more | the least common multiple |

### Orthogonal polynomials and spherical harmonics

| Function | Arguments | Meaning |
|---|---|---|
| `legendre(n, x)` `P_n(x)` | 2 | the Legendre polynomial of degree n |
| `assoc_legendre(ell, m, x)` `P_{\ell}^m(x)` | 3 | the associated Legendre function of degree `\ell` and order m, with the phase (-1)^m |
| `hermite(n, x)` `H_n(x)` | 2 | the Hermite polynomial of degree n, the physicists' (probabilists write He_n) |
| `laguerre(n, x)` `L_n(x)` | 2 | the Laguerre polynomial of degree n |
| `assoc_laguerre(n, alpha, x)` `L_n^{(\alpha)}(x)` | 3 | the generalised Laguerre polynomial of degree n and parameter `\alpha` |
| `chebyshevt(n, x)` `T_n(x)` | 2 | the Chebyshev polynomial of the first kind, of degree n |
| `chebyshevu(n, x)` `U_n(x)` | 2 | the Chebyshev polynomial of the second kind, of degree n |
| `Ynm(ell, m, theta, phi)` `Y_{\ell}^m(\theta, \phi)` | 4 | the spherical harmonic of degree `\ell` and order m, at polar angle `\theta` and azimuth `\phi` |

### Special functions

| Function | Arguments | Meaning |
|---|---|---|
| `beta(a, b)` `\mathrm{B}(a, b)` | 2 | Euler's Beta function |
| `zeta(s)` `\zeta(s)` | 1 | Riemann's zeta function |
| `Log(z)` | 1 | the principal branch of the logarithm, `\operatorname{Log}` |
| `erfc(x)` | 1 | the complementary error function, 1 - erf |
| `erfi(x)` | 1 | the imaginary error function, -i erf(ix) |
| `Ei(x)` | 1 | the exponential integral |
| `Si(x)` | 1 | the sine integral |
| `Ci(x)` | 1 | the cosine integral |
| `LambertW(x, k)` `W_k(x)` | 1 to 2 | Lambert's W, the inverse of w e^w, on branch k: 0, the principal, or -1 |
| `elliptic_k(m)` `\operatorname{K}(m)` | 1 | the complete elliptic integral of the first kind, of parameter m = k^2 |
| `elliptic_e(m)` `\operatorname{E}(m)` | 1 | the complete elliptic integral of the second kind, of parameter m = k^2 |
| `hyper(a, b, z)` `{}_pF_q(a; b; z)` | 3 | the generalised hypergeometric function; `hyper([a_1, ...], [b_1, ...], z)` |
| `rf(a, n)` `RisingFactorial` `(a)_n` | 2 | the Pochhammer symbol, the rising factorial a(a + 1)...(a + n - 1) |
| `normcdf(x)` `\Phi(x)` `N(x)` | 1 | the standard normal distribution function, (1 + erf(x/√2))/2 |
| `Heaviside(t, c)` `u_c(t)` | 1 to 2 | the unit step, 0 before c and 1 after it, and 1/2 at c |
| `DiracDelta(x)` `\operatorname{DiracDelta}(x)` | 1 | the Dirac delta, a distribution; `\delta(x)` is not read as it, as `\delta` is a variation in physics |

### Book names not supported yet

A name books write for a function no integration provides is refused, saying what it is, never read as an unknown function or a symbol: `\operatorname{Res}_{z=a} f`, the residue of a function at a point; `\operatorname{tr} A`, the trace of a matrix written as an operator; `\operatorname{rank} A`, the rank of a matrix; `\operatorname{Var}(X)`, the variance of a random variable; `\mathcal{L}\{f\}`, the Laplace transform; `\mathcal{F}\{f\}`, the Fourier transform; `\epsilon_{ijk}`, the Levi-Civita symbol.

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
| `eig` | `\lambda_k, v^a{}_k := \operatorname{eig}(A^a{}_b)`: eigenvalues and eigenvectors |
| `exp` | the exponential of a tensor named without indices: `\exp(-i H t)` |
| `inv` | the inverse; `(\Sigma^{-1})^{ab}` is written as on paper, and solved for, not formed |
| `log` | the logarithm of a matrix |
| `spec` | `\operatorname{spec}(A^a{}_b)`, the eigenvalues; with a second argument n, the n largest, or smallest if n is negative |
| `sqrt` | the square root of a matrix, `\sqrt{\Sigma}` |
| `svd` | `U_{tk}, s_k, V_{ak} := \operatorname{svd}(X_{ta})`: the singular values and vectors |
| `tr` `trace` | the trace of a map |

## LaTeX

Typesetting is ignored: spacing, sizes, `\label`, `\tag`, `$`, `$$`, `\[ \]` and the `&` of alignments. Every statement inside a display is LaTeX, however it is written. An unknown command is an error that suggests the closest one.

### Commands

| Command | Meaning |
|---|---|
| `\alpha \beta \chi \Delta \delta \epsilon \eta \Gamma \gamma \iota \kappa \Lambda \lambda \mu \nu \Omega \omega \omicron \Phi \phi \Pi \pi \Psi \psi \rho \Sigma \sigma \tau \Theta \theta \Upsilon \upsilon \Xi \xi \zeta` | Greek letters, each one symbol |
| `\varDelta \varepsilon \varGamma \varkappa \varLambda \varOmega \varPhi \varphi \varPi \varPsi \varrho \varSigma \varsigma \varTheta \vartheta \varUpsilon \varXi` | variant letterforms: each the letter after `var` |
| `\ell \hbar \infty \odot \star \varpi` | symbols; `\odot` is the Sun, as in `M_\odot`, and between tensors the elementwise product |
| `\arccos \arccot \arcsin \arctan \arg \cos \cosh \cot \coth \csc \csch \det \exp \gcd \Im \ln \log \max \min \Re \sec \sech \sgn \sin \sinh \tan \tanh` | functions, each calling the builtin of its name, except that `\arccos` calls `acos`, `\arccot` calls `acot`, `\arcsin` calls `asin`, `\arctan` calls `atan`, `\Im` calls `im`, `\Re` calls `re`, `\sgn` calls `sign`; `\sin^{-1}` is the inverse, and only `\log` takes a subscript, its base |
| `\bar \breve \check \ddot \dot \hat \overline \tilde \vec \widehat \widetilde` | accents: `\dot{a}` and `\ddot{a}` are time derivatives of a function of one variable; the others name a new symbol |
| `\ast \cdot \otimes \times` | multiplication, `*`; between tensors, their product |
| `\div` | division, `/` |
| `\lt \gt \le \leq \leqslant \ge \geq \geqslant \ne \neq` | comparisons, in this order: `<`, `>`, `<=` (three spellings), `>=` (three), `!=` (two) |
| `\longrightarrow \mapsto \rightarrow \to` | an arrow, as in `\lim_{x \to 0}` |
| `\coloneq \coloneqq \defeq \equiv \triangleq` | a definition, `:=` |
| `\lvert \mid \rvert \vert` | a bar: absolute values, norms and evaluation bars |
| `\langle \rangle` | brackets: `\langle x \rangle` groups |
| `\pm \mp` | both signs, so two values |
| `\wedge` | the wedge product |
| `\circ` | composition: read, not computed |
| `\approx \asymp \cong \doteq \gg \in \leftrightarrow \ll \longleftrightarrow \notin \propto \sim \simeq \subset \subseteq \supset \supseteq` | relations, read so that the equations around them run, but not computed; `\in` also declares index labels |
| `\iff \implies \Leftrightarrow \Longleftrightarrow \Longrightarrow \Rightarrow` | between equations, each side a statement |
| `\cdots \ddots \dots \dotsb \dotsc \dotsi \dotsm \dotso \ldots \vdots` | terms not written: read, not computed |
| `\hbox \mbox \operatorname \text \textbf \textit \textnormal \textrm \textsf \texttt \textup` | text: one name, as `\text{otherwise}`; `\operatorname{name}` names a function |
| `\bm \boldsymbol \lefteqn \mathbb \mathbf \mathbin \mathcal \mathclose \mathfrak \mathinner \mathit \mathnormal \mathop \mathopen \mathord \mathpunct \mathrel \mathrm \mathscr \mathsf \mathtt \pmb \substack` | typefaces: what they hold is read as it is, so `\mathrm{d}x` is `d x` |
| `\bf \bfseries \boldmath \cal \em \it \itshape \mdseries \mit \normalfont \rm \rmfamily \sc \sf \sl \tt \upshape` | plain TeX's typefaces, for the rest of the group: `{\rm d}x` |
| `\aftergroup \allowbreak \centering \displaybreak \displaystyle \enspace \hfill \hline \hphantom \hspace \label \limits \medspace \negmedspace \negthickspace \negthinspace \noindent \nolimits \nonumber \notag \phantom \qquad \quad \scriptscriptstyle \scriptstyle \tag \textstyle \thickspace \thinspace \vphantom \vspace` | layout, ignored; `\quad` and `\qquad` also separate equations |
| `\Big \Bigg \Biggl \Biggm \Biggr \Bigl \Bigm \Bigr \big \bigg \biggl \biggm \biggr \bigl \bigm \bigr` | delimiter sizes, ignored |
| `\linebreak \newline` | end a statement, as `\\` does |
| `\cases \matrix \pmatrix` | plain TeX's matrices and cases: `\matrix{a & b \cr c & d}` |
| `\cfrac \dfrac \frac \over \tfrac` | fractions, `\frac{a}{b}` and plain TeX's `{a \over b}`; `\frac{d}{dx}` and `\frac{\partial f}{\partial x}` are derivatives |
| `\sqrt` | roots, `\sqrt{x}` and `\sqrt[n]{x}` |
| `\binom \choose \dbinom \tbinom` | binomial coefficients, `\binom{n}{k}` and `{n \choose k}` |
| `\prod \sum` | a sum, `\sum_{n=1}^{N}`, and a product; `doit` carries out one kept as written |
| `\iiint \iint \int \oint` | integrals, single, double and triple, which end at their differentials: `\int_0^1 f \, dx` |
| `\lim` | a limit, `\lim_{x \to 0}`, or from one side, `\lim_{x \to 0^+}` |
| `\nabla \partial` | the partial and the covariant derivative, `\partial_\mu` and `\nabla_\mu`; `\partial_t^2` is the second derivative |
| `\Box \square` | the d'Alembertian, `\nabla^\mu \nabla_\mu`, as is `\nabla^2` |
| `\pounds` | the Lie derivative, `\pounds_V T_{ab}` |
| `\dv \pdv` | the physics package's derivatives: `\pdv{f}{x}`, `\pdv[2]{f}{x}`, `\dv{f}{x}` |
| `\lceil \lfloor \rceil \rfloor` | the floor `\lfloor x \rfloor` and the ceiling |
| `\left \right` | `\left( ... \right)` groups, with any delimiters; between bars it is an absolute value, and from `\left.` to a bar with `_{x=a}` an evaluation bar |
| `\begin \end` | an environment, `\begin{pmatrix} ... \end{pmatrix}` |
| `\bgroup \egroup` | plain TeX's braces |
| `\cr` | ends a row of plain TeX's `\matrix` and `\cases` |
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

- `\abs`: the physics package's absolute value, is not supported; write `|x|`, or `\left| x \right|`.
- `\bmod \mod \pmod`: the remainder is not supported yet; write `x - n \lfloor x/n \rfloor`.
- `\bra \braket \ket`: dirac's notation is not supported; write a state as a list of its components.
- `\dagger`: the Hermitian conjugate is not supported yet; write `conjugate(A_{ba})`.
- `\dd`: the physics package's differential, is not supported; write `d x`, or `\mathrm{d} x`.
- `\def \let \newcommand \providecommand \renewcommand`: defines a macro, which a program cannot; write what it stands for; relativisticpy check reads the macros of a document.
- `\eval`: the physics package's evaluation bar, is not supported; write `\left. f \right|_{x=0}`.
- `\exists`: the quantifier is not supported; write `\text{there is}`.
- `\forall`: the quantifier is not supported; write `\text{for all}`.
- `\inf`: the infimum is not supported; write `\min`, or the value.
- `\intercal \top`: the transpose is not supported yet; the transpose of `A_{ab}` is `A_{ba}`.
- `\liminf \limsup`: is not supported; write `\lim_{x \to a}` for a limit.
- `\norm`: the physics package's norm, is not supported; write `\| x \|`.
- `\not`: is not supported; write `\neq` for not equal, or `\notin`.
- `\order`: the physics package's order, is not supported; write the terms out, or `\mathcal{O}(x^2)` as a symbol.
- `\qty`: the physics package's brackets, is not supported; write `\left( ... \right)`.
- `\sup`: the supremum is not supported; write `\max`, or the value.
- `\Tr`: the physics package's trace, is not supported; write `\operatorname{Tr}(A)`, or contract the indices: `A^{a}_{a}`.
- `\tr`: the physics package's trace, is not supported; write `\operatorname{tr}(A)`, or contract the indices: `A^{a}_{a}`.
- `\va`: the physics package's vector arrow, is not supported; write `\vec{v}`.
- `\vb`: the physics package's bold vector, is not supported; write `\mathbf{v}`.

### Unicode

Unicode reads as the LaTeX it stands for. A Greek letter, `θ`, is its command, and so are ∂ `\partial`, ∇ `\nabla`, √ `\sqrt`, ∑ `\sum`, ∏ `\prod`, ∫ `\int`, ∬ `\iint`, ∭ `\iiint`, ∮ `\oint`. The operators · ⋅ × ÷ − ≤ ≥ ≠ → ↦ ≡ ≔ ′ ± ∓ ∧ ∘ ⊗ … ⋯ and the relations ∼ ≃ ≈ ≅ ∝ ≪ ≫ ∈ ∉ ⊂ ⊆ ↔ are LaTeX's. Superscripts and subscripts are scripts, so `r²` is `r^{2}`, and the double-struck letters, ℝ ℂ 𝔼 𝟙, are `\mathbb`'s. A combining accent, as in ȧ, is an error that names it: write the command, `\dot{a}`.

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
