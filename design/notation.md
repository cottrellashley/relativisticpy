# Notation decisions

How the language reads notation that textbooks write in more than one way, adopted on 21 November 2026 from the curriculum's roadmap (`curriculum/ROADMAP.md`, section 6). The rule behind every one is the interface's: read by the operands, take the most likely reading, and say so in a note wherever another reading was possible. A user's definition always wins.


These are the places where textbooks disagree, or a reading is ambiguous, and where the language today
guesses, usually silently. Each names the capabilities that wait on it. The recommendations follow
the interface's rule: never guess silently. Where a reading is chosen, say so in a note.

## 1. What a parameter nothing declares is

**Waiting on it:** `default-assumptions` (27 cases), `inequality-assumptions` (4).

**Why it is open.** SymPy takes every symbol complex, so integrals come back as Piecewise conditions with `polar_lift`, and |f|² and conjugates never simplify. Books never say that a width, a mass, ħ, a temperature or a frequency is positive, or that a time or an angle is real.

**Recommendation.** Keep symbols general. When a result would be a Piecewise of conditions, or a modulus or conjugate that a sign or reality assumption removes, give the value under that assumption and state it in a note and in the result's data ("assuming a > 0"). Declarations override. Never assume against a declaration. The coordinates of a metric and the variable of an integral over real limits are real. Complex analysis keeps z complex because nothing there needs otherwise.

## 2. A relation on a lone name: assumption or claim?

**Waiting on it:** `inequality-assumptions`: `a > 0`, `a, b > 0`, `t > 0`, `x \in \mathbb{R}`.

**Why it is open.** Today `a > 0` is printed and assumes nothing. `a, b > 0` splits at the comma into `a` and `b > 0`. `x \in \mathbb{R}` is refused, and `x \in \mathbb{R}^3` declares x an index (the regression the review recorded).

**Recommendation.** A relation between undefined names and a constant, standing alone, is an assumption, distributed over a list of names, with a note. `\in \mathbb{R}` (ℂ, ℤ, ℕ, no exponent) is an assumption. `\in \mathbb{R}^n` declares a space only for names used as index labels.

## 3. When `=` between expressions is a claim, an equation or a constraint

**Waiting on it:** `decide-identities` (7), `implicit-relations` (4), `decide-inequalities` (3), `order-terms` (2).

**Why it is open.** Today such an `=` is an equation to solve, shown as `Eq` unless a quick zero test proves it. A later `dy/dx` ignores it. Curriculum cases write identities (the Black–Scholes PDE, B(ζ₁)B(ζ₂) = B(ζ₁+ζ₂)) and constraints (x³ + y³ = 6xy) with the same sign.

**Recommendation.** Decide every relation whose sides are fully known. It is True when it holds identically, by simplification, expansion, and then numerical evidence under a verdict of its own ("holds numerically"); otherwise it stays an equation, "not an identity". An equation between undefined names that a later derivative or integral uses makes them dependent (y an implicit function of x), with a note. An inequality claim is decided for all real values of its free names.

## 4. Should sums with a closed form be computed?

**Waiting on it:** `closed-form-sums` (6), `geometric-exponential-sums` (2), `sum-index-integer` (2).

**Why it is open.** By design, sums with symbolic or infinite limits are kept as written until `doit`, while finite numeric ones are computed. The review found `\sum_{n=1}^{3} n = 6` judged as differing. Every book means the value.

**Recommendation.** Compute a sum or product when a closed form is found within a time budget. Otherwise keep it as written, with a note. A summation index is an integer.

## 5. What a subscript means

**Waiting on it:** `label-subscript` (15), `subscripted-function` (12), `family` (11), `unbraced-index` (4), `list-subscript` (3), and five more.

**Why it is open.** It can be part of a name (r_s, λ_1, σ_x, R_⊙, r_+), an index (T_{μν}), a family parameter (c_n, ψ_n(x)), an element (x_i of a list), a partial derivative (f_{xy} in Stewart), a list of labels (j_{ν,k}) or the length of a Pochhammer symbol ((a)_n). Today Greek and built-in bases with a label become components (`\lambda_1 :=` is refused, R_⊙ is the Ricci tensor), `\psi_1(x)` is a product, and `c_1` after `c_n` is a new symbol.

**Recommendation.** Read it in this order, and say which reading was taken where another was possible:

1. a declared index label, or an index of a tensor in force, is an index, braced or not;
2. on the left of `:=`, a one-letter subscript that occurs on the right makes a family, so `c_1`, `a_{n+1}` and `\psi_3(x)` evaluate it;
3. on a list, it is an element, counted from the lower limit of the sum that ranges over the whole list (decision 17);
4. after a parenthesised expression it is a Pochhammer length; but after a partial derivative in brackets, by a variable of its function, it is another derivative, as Stewart writes `(f_x)_y` for f_{xy} (23 November 2026: it was the Pochhammer symbol of f_x);
5. otherwise it is part of the name, whatever the base: Greek letters, numbers, signs, ⊙. A built-in tensor claims a subscripted name only while a metric is in force;
6. `f_x` is a partial derivative only on a defined function of x, with nothing defining `f_x`; with brackets after it, `f_x(1, 2)` and `f_\theta(r, \theta)`, it is the partial derivative at the point they hold, as Stewart writes it, a Latin or a Greek variable alike (23 November 2026: `f_\theta(r, \theta)` was the derivative times the pair, and `f_x(x, y)` an unknown function), and at a point of another number of coordinates it is refused;
7. in a letter run the subscript binds to the last letter, as in TeX (`iE_1` is i·E_1);
8. a comma separates labels on a name that is not a tensor.

**What implementing one resolver settled** (23 November 2026). One resolver reads every name with a subscript, and every name with brackets right after it, and each layer of the compiler asks it (`relativisticpy.reading.subscripts`); before, each layer had a test of its own, and the order of the layers decided. A user's definition always wins, so the readings of what the program defines come before those of what is built in. In order: a name the program defines as it is written, as `c_1` after `c_1 := 5`; an index of a tensor the program defines, braced or not; an entry of a matrix, `A_{12}`; a member of a family, or of a recurrence; an element of a list; a member of a sample; a sequence at a point; a partial derivative; an index of a built-in tensor in force; a book name of an integer (decisions 34 and 35); and, last, a label, part of the name. Where the program defines a base as two of these, as a family `c_n` and a list `c`, the first is taken, whatever the letter of the subscript, and a note says which the other was. Whether a subscript picks a member is decided on the subscript as the parser read it, a number, a name or an expression, never on its text. So `c_{2k}` is the member at 2k, while letters side by side, as `\mu\nu`, and a label before a number, as `\mathrm{A}0`, are labels. The label of a family may be written only in the name of a function its value calls, as n is in `y_n(x) := P_n(x)`; and a name the program defines with a label owns its letter with any subscript, so that `y_2(x)` is then the program's, not the spherical Bessel function.

A sum or a product whose index runs through a subscript, as `\sum_{n=0}^{3} a_n` and `\sum_{k=1}^{3} c_{2k}`, is taken term by term where its limits are whole numbers known as the program is read, each name read at the value the index has there as it is read written out, so that the sum agrees with its terms: the member the program gives as a name of its own, the family at the subscript, an element of a list counted from the lower limit (decision 17), or a name of its own, `a_1 + a_2 + a_3`. With other limits a family is summed by its value, unless the sum runs past a member the program gives apart from it, which is refused; and a name nothing defines is refused, never taken for one constant in every term. Under a sum over n, `H_n` and `B_n` are names for each term even after `n \in \mathbb{N}`, as the coefficients of a series are (decisions 34 and 35), and a note says so.

## 6. What a superscript means

**Waiting on it:** `transpose-dagger` (10), `superscript-label` (8), `star-conjugate` (6), `nth-derivative` (2), and four more.

**Why it is open.** It can be a power, an index, a label (Y_l^m, E^{(0)}), a derivative order (f^{(4)}), a transpose (A^T), a conjugate (ψ^*) or part of a name (r^*, the tortoise coordinate). Today `Q^T` is refused as a power, `\sigma^2 :=` as a component, and `\psi^*` is a power by a symbol named `*`.

**Recommendation.** Read it as follows:

1. on the left of `:=` a superscript is part of the name being defined (`\sigma^2`, `\langle x^2 \rangle`, `E_n^{(0)}`), with a note;
2. `^{(n)}` on a function is its nth derivative, and on a new name an order label;
3. `^T`, `^\top` and `^\intercal` on a matrix are its transpose when T has no value, and `^\dagger` is always the conjugate transpose;
4. `^*` after a call or a defined complex quantity is the conjugate, and on a new name part of the name;
5. on a special function's name it is a label;
6. `e^{...}` is always the exponential;
7. a two-index tensor's name alone is its trace, as R is, so K² is (K^μ_μ)².

**A superscript that is no exponent** (23 November 2026). On a name nothing defines, a superscript that can only be an index is one, never a power: a lone 0, as in J^0 and p^0; digits that are no number as numbers are written, as in \eta^{00} and F^{01}; and index letters beside a 0 or before a digit, as in F^{0i}, F^{\mu 0} and F^{\mu 1}. It names a component, a name of its own, as J_0 does, with a note; for the power, brackets write it, `(J)^0`. A letter of such a superscript that the statement also writes as a subscript, as i in `\partial_i F^{0i}`, is an index the statement contracts, which needs the name to be a tensor: it is refused until it is defined, as `\partial_i F^{i}` is. Numbers written as numbers are, `x^{10}` and `x^2`, are exponents, and so is a number before letters, as `x^{2n}`. J^0 was 1, so that Carroll's (1.76), `\partial_i F^{0i} = 4\pi J^0`, differed by −4π. A letter that a parameter or a sum gives a value where it is read, as the parameter of a family or a function, or the index of a sum, is never an index (a value of the program's own, `\lambda := 2`, leaves a label an index): in `a_{n} := \frac{x^n}{n!}` the subscript names the family's parameter, and x^n is a power, where it was refused as an index on x.

## 7. What a prime means

**Waiting on it:** `primed-names` (11).

**Why it is open.** It can be a derivative or another frame (x', t', E', F'^{μν}, k'^μ, μ'). Today a prime is always a derivative, so `x' :=` is refused and `\mathbf{r}(t') :=` silently becomes an equation.

**Recommendation.** A prime is a derivative only on a function of one variable: one the program defined, or an unknown function written with its argument. Elsewhere, and always on the left of `:=`, it is part of the name. On an index it marks another frame's index.

**Primes after brackets** (23 November 2026). `(x^3)'` was x'³, the prime put on the base inside the brackets. A prime after brackets is the derivative of all they hold, by the one variable it is written with, as `f'` is of a function of one variable, with a note that names the variable: `(e^{-x^2})'` is −2x e^{−x²}, `(x^2 + 1)'(3)` is 6, and `(3)'` is 0. What is written with more than one variable, as `(x^2 y)'`, is refused, with a hint to write `\frac{d}{dx}`. Braces do not show, so `{x^3}'` stays x'³, as TeX sets it.

## 8. What accents mean

**Waiting on it:** `unit-vectors` (8), `accent-script` (7), `conjugate-bar` (3), `lagrangian-velocities` (3), `accented-definitions` (1).

**Why it is open.** \bar can be a conjugate (complex analysis, \bar m^ρ), a mean (statistics) or the Dirac adjoint. \hat can be a unit vector, an operator or an estimator. \dot can be a time derivative or a generalised velocity. Today they are new names: `\bar z` is zbar, `\hat{\mathbf{x}}` is xhat, and `\dot\theta^2` is the dot of θ².

**Recommendation.** Read each by what it sits on and where:

1. `\bar` of a defined complex quantity, or of z in a complex expression, is the conjugate, and ∂/∂\bar z the Wirtinger derivative;
2. an accented name on the left of `:=` is a new name (`\bar x :=` the mean);
3. `\hat x` and `\hat{\mathbf{x}}`, with x a coordinate, are unit vectors, while `\hat p :=` names an operator;
4. `\dot q` is independent of q in ∂L/∂\dot q and is dq/dt under d/dt, and a script after an accent applies to the accented name.

## 9. Do typefaces make different names?

**Waiting on it:** `typeface-names` (4).

**Why it is open.** Today `\mathbf{r}` is r and `\mathcal{F}` is F, so `\mathbf{N} :=` redefines the builtin N, and `\mathcal{F}^{\alpha\beta} := \ldots` overwrites F. Mechanics writes \mathbf{r} and r = |\mathbf{r}| on one line.

**Recommendation.** Yes, a typeface makes a different name. `\mathcal{L}\{\ldots\}`, with braces, is the Laplace transform; `\mathcal{L} :=` is a Lagrangian density.

## 10. Is LaTeX decided per statement or per program?

**Waiting on it:** `latex-per-statement` (3).

**Why it is open.** Today a statement without a backslash in a LaTeX program is read as plain text, so `6xy` is the name `xy`, `-ka` is `ka`, and `e^{At}` makes `At` one name. The review recorded this; three cases here, and two variants written for this roadmap, met it.

**Recommendation.** A program given as LaTeX (a `latex` fence, a document, `--latex`) is LaTeX in every statement.

*The curriculum's fences* (23 November 2026). The curriculum gave a case's `latex` fence to the workbook as text, which decides each statement by its own commands: Black–Scholes' last line, `S_0 N(d_1) - K e^{-r T} N(d_2)`, has none, so N was numerical evaluation, as in plain text, and the call was worth 8.42, silently. A `latex` fence is now read as LaTeX in every statement, so that N is a function of the program's own (decision 15), and a result that holds it at a number says so (decision 26). Text given to a workbook, with no fence, is still decided a statement at a time: plain text and LaTeX mix (docs/latex.md).

## 11. How far an unbracketed function argument runs

**Waiting on it:** `function-argument-extent` (2).

**Why it is open.** Today `\sin (2k-1) x` is x·sin(2k − 1), and `\sin\theta \frac{\partial Y}{\partial\theta}` is sin(θ·∂Y/∂θ).

**Recommendation.** Follow the books: the argument is the following product of factors, up to an operator, a fraction, a derivative or another function. So `\sin (2k-1)x` is sin((2k−1)x) and `\sin\theta\,\frac{a}{b}` is sin θ·a/b. Give a note where both readings are plausible.

## 12. The identity, and numbers beside matrices

**Waiting on it:** `identity-matrix` (7).

**Why it is open.** Books write `A - \lambda I`, `\mathbf{1}`, `\mathbb{1}`, and the Dirac `(\gamma^0 E - \gamma^3 p - m)`. Today `I` is a plain symbol (the imaginary unit is `i`), and a number added to a matrix is refused.

**Recommendation.** An undefined `I`, `\mathbf{1}`, `\mathbb{1}` or 𝟙 in matrix arithmetic is the identity of the matching size, with a note; a defined `I` keeps its value. A number added to a square matrix inside a matrix equation is that multiple of the identity, with a note. Where `\mathbf{1}` meets a vector rather than a matrix, it is the vector of ones (decision 23).

## 13. Brackets

**Waiting on it:** `dirac-notation` (9), `operators` (7), `poisson-bracket` (3), `anticommutator`, `user-notation`, `kl-divergence`.

**Why it is open.** `\{f, g\}` can be a Poisson bracket, an anticommutator or a set; `[A, B]` a commutator, a list or an interval; `\langle\cdot\rangle` an expectation value, an inner product or Dirac notation; `\|` a norm or the separator in D(p‖q). Today `\{L_x, L_y\}` is a tuple, so a true Poisson bracket is judged False.

**Recommendation.** Decide by the operands:

1. `\{f, g\}` of scalar functions is a Poisson bracket, with canonical pairs by name (x with p_x) or given as `_{q,p}`; of matrices or operators it is an anticommutator; otherwise a set;
2. `[A, B]` of operators or matrices is a commutator;
3. `\langle\ldots\rangle` with a bar is Dirac notation; without, an expectation value or a user's bracket (`\langle p, q\rangle :=`);
4. `D(p \| q)` is the relative entropy, in bits (Cover and Thomas).

## 14. What `\sim` and `\approx` claim

**Waiting on it:** `random-variables` (5), `asymptotic-relations` (3), `matrix-similarity`, `periodic-identification`, `perturbative-ansatz`.

**Why it is open.** `\sim` can be asymptotic equivalence (Stirling), a scaling law (m ∼ (T_c − T)^β), 'distributed as' (X ∼ Poisson(λ)), similarity of matrices, or an identification (τ ∼ τ + β). Today it is read and not computed.

**Recommendation.** Decide by the operands:

1. a random variable and a distribution: distributed as;
2. two matrices: similar;
3. a coordinate and itself plus a constant: an identification;
4. two functions with an implied limit: asymptotic equivalence (lim f/g = 1);
5. a power law with an unknown exponent: a scaling law, solved for the exponent;
6. anything else: refused.

`\approx` with unknown parameters is an ansatz to solve order by order; otherwise it stays unchecked.

**What implementing the first reading settled** (22 November 2026). `X \sim D`, where D has a form a distribution declares, declares X for the rest of the program, as an assumption does; between anything else `\sim` stays refused, as before. A form says what each argument is: `\mathcal{N}(\mu, \sigma^2)` and `N(\mu, \sigma^2)` take the variance, `\Gamma(k, \theta)` the shape and the scale, `\operatorname{Exp}(\lambda)` the rate, and `\operatorname{Geom}(p)` counts the trials up to the first success; a distribution called by its name in plain text, `Normal(mu, sigma)`, takes SymPy's and SciPy's parameters, the standard deviation. The note of each declaration gives the value of each parameter. `\mid` outside brackets is the conditional bar, and `X \mid N \sim D` a law given another. Random variables declared apart are taken independent, with a note, and `\overset{\text{iid}}{\sim}` or `\text{independent}` says so. Inside a probability, an expectation, a variance or a covariance, a capital letter nothing defines is a random variable of a law not given, as probability books write one; P and E are a probability and an expectation only of random variables, but for an upright E before square brackets, `\operatorname{E}[X]`, which is always the expectation, as `\operatorname{E}(m)` stays the elliptic integral. A probability that holds only for some values of a parameter the program says nothing of is given where the parameter lies among the values the variable takes, with a note, as books give P(T > t) = e^{-λt} for t ≥ 0. A member of a sample, declared as `X_1, \ldots, X_n \sim D`, is written with any subscript that is written as an integer is, whole numbers and names in sums, differences and products, as `X_1`, `X_k` and `X_{k+1}`; any other subscript on the sample's letter is refused (23 November 2026).

## 15. Book names that collide with the language's or with each other

**Waiting on it:** `special-functions` (18), `step-delta` (6), `normal-cdf` (2).

**Why it is open.** N(x) is the normal distribution function in Hull and numerical evaluation in the language; Black–Scholes silently gives 8.42 instead of 4.76. Φ is the normal distribution function or a potential. Γ is Euler's gamma or a Christoffel symbol, and H_n the physicists' or the probabilists' Hermite polynomial. u_c(t) is a step or a velocity, δ_0 a phase shift or the Kronecker delta, π_j a vector or the constant.

**Recommendation.** A user's definition always wins. Otherwise:

1. `\Gamma(x)` with one bracketed argument and no indices is Euler's gamma; with indices it stays Christoffel;
2. `\mathrm{B}(a, b)` is the Beta function, and `H_n` the physicists' Hermite polynomial (probabilists write He_n);
3. `u_c(t)` is the unit step where u is not defined;
4. `N(x)` and `\Phi(x)` are the standard normal distribution function, with a note. This retires `N` as numerical evaluation in LaTeX, where `evalf` and `numerical` remain. *Amended (23 November 2026):* `\Phi(x)` alone is. `N(\pi)` was Φ(π) where a program meant π as a decimal, and `N(\pi, 5)` an unknown function: a written form under the name of another capability, as `N` is numerical evaluation's, takes that capability's calls from it, and the conformance suite now refuses one. In LaTeX `N(x)` is a function of the program's own, as the lapse N(t) is, and `evalf` and `numerical` evaluate numerically; after `\sim`, `N(\mu, \sigma^2)` stays the normal distribution;
5. a name with one index (`\delta_0`, `\pi_j`) is not the two-index Kronecker delta nor the constant.

## 16. Index notation without a spacetime

**Waiting on it:** `index-ranges` (6), `levi-civita` (5), `metric-without-coordinates` (5), `unknown-vectors` (2), `cartesian-indices`, `embedding-indices`, `definition-index-check`.

**Why it is open.** `\eta_{\mu\nu} = \operatorname{diag}(-1, 1, 1, 1)` needs coordinates today. Under a declared Euclidean metric, repeated lower indices are refused. `i, j = 1, 2, 3` is a statement 'read but not computed'. The sign of ε^{0123} is +1 in Jackson and in this curriculum, and −1 in Peskin; MTW and Carroll fix ε_{0123} = +√|g|. A definition relabels free indices by position.

**Recommendation.** Apply these rules:

1. accept a metric given by components alone;
2. sum repeated lower indices under a Euclidean metric (Cartesian tensors), and check free indices on both sides of every definition;
3. read `i, j = 1, 2, 3` and `n = 0, \ldots, N-1` as declaring a range;
4. default ε to Carroll's convention, and let a statement such as `\epsilon^{0123} = 1` set it, with a note;
5. give an index whose components outnumber the coordinates a space of that size;
6. read an undefined name with declared indices as an unknown vector, not a scalar.

## 17. Where a list's positions start

**Waiting on it:** `list-subscript` (3).

**Why it is open.** A sample mean writes `\sum_{i=1}^{n} x_i`; a DFT writes `\sum_{n=0}^{N-1} x_n`. Both on a plain list.

**Recommendation.** Take the base from the sum when it ranges over the whole list. Otherwise refuse, with a hint, rather than guess.

## 18. When d is a differential

**Waiting on it:** `d-variable` (2).

**Why it is open.** Today `d^2` with nothing after it is ∂², and `D^2` is ∇², because D is the covariant derivative. So a current loop of diameter d is refused.

**Recommendation.** d is a differential only before a variable, inside an integral, a derivative or a line element. Anywhere else, and in an evaluation bar's list of values, it is a variable. D is the covariant derivative only with an index.

**Euler's D** (23 November 2026). Stewart writes the derivative `Df(x)` and `D_x f(x)`; both were read as products, D times f(x) and a symbol D_x times f(x). `D_x`, which the lexer joins into one name, is read as `D_{x}` is wherever it stands before what it acts on: the covariant derivative with the index x, which, where there are no coordinates, is the derivative by the variable x, with a note. Alone, as in `D_x + 1`, it is a name. A bare D that nothing defines, written against a function of one variable applied, `D f(x)` or `Df(x)`, is Euler's operator, the derivative f'(x), with a note that says how to write the product, `D \cdot f(x)`. D with a value, D before anything but a function applied, as in Fick's `D \frac{\partial^2 c}{\partial x^2}`, and D before a function of several variables multiply, as before.

## 19. Ellipses and index sets given by a condition

**Waiting on it:** `conditional-index-sets` (2), `ellipsis-patterns` (1).

**Why it is open.** Books write 3·5·7⋯(2n+1), `\prod_{p \text{ prime}}` and `\sum_{k \neq 0}`.

**Recommendation.** Infer an arithmetic progression that ends in a general term, with a note, and refuse any other ellipsis. Accept a condition under a sum or product as its index set.

## 20. Operators as values, and rules written with `=`

**Waiting on it:** `operators` (7), `operator-algebra` (3).

**Why it is open.** Books write `\hat p := -i\hbar \frac{d}{dx}`, `[\hat x, \hat p] = i\hbar` and `a_-|n\rangle = \sqrt{n}|n-1\rangle`. Writing each operator as a function of its operand, `Lx(g) := ...`, already makes three cases pass.

**Recommendation.** A definition whose right side is a differential operator defines an operator, applied by juxtaposition and composed by products. An `=` whose left side is a pattern in undefined operators or kets defines a rule for what follows, not a claim.

## 21. Transform conventions

**Waiting on it:** `laplace-fourier` (5).

**Why it is open.** ℒ maps t to s by convention. Fourier transforms differ: Bracewell writes e^{−2πixs}, physics e^{−iωt}. `\supset` (Bracewell's pair) is also the superset.

**Recommendation.** Name the variables by convention (t → s for ℒ; x → s or t → ω for ℱ, by the letters used). Take the Fourier convention from a declaration, or from the 2π in a written integral. `f \supset F` between functions is a Fourier pair.

## 22. Quantified and prose claims

**Waiting on it:** `quantifiers` (6, all level 5).

**Why it is open.** ∀, ∃, 'converges uniformly on' and images of sets are refused today.

**Recommendation.** Keep them refused until a design exists. The first step worth taking is 'for all x in a set' as a claim decided as identities and inequalities are.

## 23. The vector of ones

**Added** on 22 November 2026, from the domains of optimization, statistics and graphs.

**Waiting on it:** `ones-vector` (6).

**Why it is open.** In optimization, statistics and finance, a bold one is the vector of ones: `\mathbf{1}^T w = 1`, `\mathbf{1}^T \Sigma^{-1} \mathbf{1}`, the degrees `A\mathbf{1}`. Decision 12 makes it the identity.

**Recommendation.** Decide by the operands, as decision 13 does for brackets. `\mathbf{1}` is the identity where it is added to or subtracted from a square matrix, as in `A - \lambda\mathbf{1}`. Everywhere else it is the vector of ones, sized by what it meets: transposed, in a product with a matrix or a vector, and in `\mathbf{1}\mathbf{1}^T`, the matrix of ones. `\mathbb{1}` and 𝟙 stay the identity; with a set as a subscript, `\mathbb{1}_A(x)` is the indicator function. Either way, say which in a note.

## 24. The imaginary unit j

**Added** on 22 November 2026, from signal processing and control.

**Waiting on it:** `imaginary-j` (8).

**Why it is open.** Engineers write j for √−1, as in `H(e^{j\omega})` and `e^{-j\omega n}`, where mathematicians write i. Elsewhere j is an index, a current density or a quantum number.

**Recommendation.** j is a name. A program that uses j for √−1 says so once, as engineering books do in their conventions: `j := \sqrt{-1}`. Where a result holds an undefined j in an exponent beside i-free mathematics, a note suggests the declaration.

## 25. ∇² and ∇ of a function of a vector

**Added** on 22 November 2026, from optimization and machine learning.

**Waiting on it:** `hessian-notation` (3).

**Why it is open.** In physics `\nabla^2 f` is the Laplacian, a scalar. In optimization it is the Hessian matrix of f, and `\nabla f(a)` is the gradient at the point a.

**Recommendation.** Decide by the function's argument. A function of one vector, `f(x)` with x in ℝⁿ, has a gradient vector and a Hessian matrix, and `\nabla f(a)` is the gradient at a. A function of coordinates, or of several scalar variables, has the Laplacian. `\Delta f` is always the Laplacian, and `\operatorname{Hess} f` always the Hessian. Say which in a note.

## 26. Functions that books define as they go

**Added** on 22 November 2026, from machine learning and signal processing.

**Waiting on it:** `ml-functions` (2), and part of `special-functions`.

**Why it is open.** σ(x) is the logistic function in machine learning, but σ is a standard deviation or a cross-section elsewhere. Books disagree on whether sinc x is sin(x)/x or sin(πx)/(πx). ReLU, softmax and fl(x) are defined where they are used.

**Recommendation.** A name with one meaning wherever it is written, as ReLU, softplus and softmax have, is a book name, provided by an integration. σ and sinc are not book names, since books disagree on them. A program defines them, as the book does where it introduces them: `\sigma(a) := \frac{1}{1 + e^{-a}}`. Calling one that is not defined leaves an unknown function, and a result that holds one carries a note saying so, so that it is never mistaken for a value. *(23 November 2026)* A result that holds such a function at a number, as N(0.77) of Hull's N(d_1), says so, whatever the function's name and however its argument is written.

## 27. `*` between signals

**Added** on 22 November 2026, from signal processing. **Adopted** on 22 November 2026, as recommended below.

**Waiting on it:** `convolution-operator` (2).

**Why it is open.** In signal processing, `x[n] * h[n]` and `f(t) * g(t)` are convolutions. Elsewhere `*` is multiplication, in plain text above all, and `z^*` is a conjugate.

**Recommendation.** `\ast` and `*` between two applied functions of the same free variable, or two sequences of the same index, are convolution, with a note. Between anything else, `*` is multiplication. `\circledast` is circular convolution, of the length N its subscript gives, `\circledast_N`, or else of the sequences' own. The cost is that `f(t) * g(t)`, meant as a product in plain text, becomes a convolution. That is why this waited for a decision.

## 28. A variable followed by brackets

**Added** on 22 November 2026, from dynamical systems.

**Waiting on it:** `call-or-product` (1, and more where systems of ODEs are written).

**Why it is open.** `\dot x = x(3 - x - 2y)` multiplies x by a bracket. It does not call a function x. Today an undefined name followed by brackets is a call.

**Recommendation.** A name followed by brackets is a call when the name is a function: defined as one, declared as one, or written elsewhere only with arguments. It is a product when the name is used as a variable in the same statement or in an earlier one, as x is inside its own bracket here. Say which in a note when the name is undefined.

## 29. `\log` without a base

**Added** on 22 November 2026, from chemistry.

**Waiting on it:** `builtin-redefinition` (1).

**Why it is open.** Mathematics and physics write log for the natural logarithm. Chemistry and engineering often mean log₁₀, as in pH and the Nernst equation.

**Recommendation.** A bare `\log` is the natural logarithm. A program that means base 10 writes `\log_{10}`, or says so once: `\log x := \log_{10} x`. That redefinition must work: a user's definition wins, and inside its own body `\log_{10}` still means the built-in.

## 30. Congruences and remainders

**Added** on 22 November 2026, from number theory.

**Waiting on it:** `congruences` (3), `remainder` (2).

**Why it is open.** `a \equiv b \pmod{m}` is a congruence, but `\equiv` alone defines (decision 3).

**Recommendation.** `\equiv` with `\pmod{m}` or `\bmod m` on its right is a congruence: a claim, decided by the remainder of a − b. `a \bmod m` is the remainder, and `a^{-1} \bmod m` is the inverse modulo m. `\equiv` without a modulus still defines.

## 31. Sequences written with square brackets

**Added** on 22 November 2026, from signal processing.

**Waiting on it:** `bracket-sequences` (9).

**Why it is open.** Oppenheim and Schafer write discrete signals as `x[n]`, `\delta[n]` and `u[n]`. In the language, square brackets after a name index a list or a tensor.

**Recommendation.** `x[n] := \ldots` defines a sequence: a function of an integer. `x[n - 1]` evaluates it, and `y[n] = \sum_k x[k] h[n - k]` is an equation of sequences. An undefined `\delta[n]` is the unit impulse and `u[n]` the unit step, with a note. Brackets after a list or a tensor still index it.

**What the review of the reading stack settled** (23 November 2026). Brackets right after a list are its element in a sum or a product over all of it, counted from the lower limit (decision 17), so that `\sum_{n=0}^{3} x[n]` of a list of four is their sum; anywhere else they are refused, as its positions have no start there, and a bracket that holds no point, as `x[1 + \epsilon]`, multiplies as before. Right after the letter of a family they are its member, as `x[n]` after `x_n := \ldots`; and a sequence written with its point as a subscript, `x_n` after `x[n] := \ldots`, is the sequence at the point, with a note. So a sequence and a family of one letter are read alike, however a member is written. `E[X]` and `P[A]` of random variables are an expectation and a probability (decision 14), whatever the dialect.

## 32. The letters of arithmetic functions: φ, μ, Λ, σ_k, ω, Ω, d and p

**Added** on 22 November 2026, from number theory.

**Waiting on it:** `arithmetic-functions` (3).

**Why it is open.** Number theory writes Euler's totient φ(n), the Möbius function μ(n), von Mangoldt's Λ(n), the divisor functions σ_k(n) and d(n), the numbers of prime factors ω(n) and Ω(n), and the partitions p(n). Physics writes the same letters for an angle or a field φ(x), a mean or a reduced mass μ, the cosmological constant Λ, a frequency ω(k), a momentum p(t), and d for a differential. Today `\varphi(36)` is an unknown function at 36.

**Recommendation.** Decide by the argument. A user's definition always wins. Otherwise the letter is the arithmetic function only where its argument is an integer: a whole number above 0, as in `\varphi(36)`, a name the program assumes an integer (`n \in \mathbb{N}`, `n \in \mathbb{Z}`), the index of a sum or a product, as d is in `\sum_{d \mid n} \mu(d)`, or a sum, difference, product or power of these, as `p(5k + 4)`. A real or unknown argument keeps the letter what it was: `\varphi(x)`, `\mu(t)` and `p(t)` stay functions of the program's. Each reading says so in a note. A number that is no positive integer, as `\phi(0)`, is no argument of these functions, and is read as before. σ without a subscript is not a book name (decision 26). A call of one of them whose argument is not an integer prints by its name, `\operatorname{totient}(x)`, which reads back as it.

**What the review of the reading stack settled** (23 November 2026). ω, Ω, μ, Λ and φ, which physics writes for a frequency, a solid angle, a mass, a constant and an angle, follow the conservative rule of decision 33. Each is its arithmetic function only of a whole number above 0 or of one name that is an integer, by assumption or as the index of a sum or a product, as `\omega(30)`, `\varphi(n)` and `\sum_{d \mid n} \mu(d)`; and never where the letter stands right after another symbol, as a factor does. So `\hbar\omega(2n + 1)` is ħ times ω times 2n + 1, with a note, whatever was written before it, and `\hbar\omega(k)` a function ω of k. d and p keep the rule above, as `p(5k + 4)`. The note of each reading says what the function is of the argument as it is written.

## 33. π(x): the prime-counting function or π times x

**Added** on 22 November 2026, from number theory.

**Waiting on it:** `prime-number-theorem` (1, at level 5).

**Why it is open.** Number theory writes π(x) for the number of primes up to x. Physics writes `\pi(r_+^2 + a^2)` and `2\pi(1 + z)`, π times a bracket, and π(x) must never stop meaning π times x there.

**Recommendation.** Decide conservatively. π is a constant, so `\pi(\ldots)` is π times what is in the brackets, as before, unless the brackets hold one name that the program assumes an integer, as n after `n \in \mathbb{N}`, which is the prime-counting function, with a note. A number is not enough: `\pi(10)` is 10π, with a note that `primepi(10)` counts the primes up to 10. Neither is the index of a sum, so `\sum_{n=1}^{N} \pi(n)` stays π times n.

## 34. H_n: the harmonic number or Hermite's polynomial

**Added** on 22 November 2026, from number theory.

**Waiting on it:** no case yet; books of analysis and combinatorics write `H_n` throughout.

**Why it is open.** Concrete Mathematics writes H_n for the harmonic number 1 + 1/2 + ... + 1/n, and H_n(x) is Hermite's polynomial (decision 15). H_0 is Hubble's constant in cosmology and the unperturbed Hamiltonian in quantum mechanics, and H_1 a perturbation.

**Recommendation.** Decide by the argument. `H_n(x)`, called, is Hermite's polynomial, as decision 15 says. `H_n` alone is the harmonic number only where its subscript is written with names the program assumes integers, as `H_n` and `H_{n+1}` after `n \in \mathbb{N}`. A number in the subscript keeps it part of the name (decision 5), so that `H_0` stays Hubble's constant; so does the index of a sum, and a name the program gives the letter with a subscript itself, as `H_1 := \ldots`. `harmonic(n)` is always the harmonic number.

## 35. B_n: the Bernoulli number or the Bell number

**Added** on 22 November 2026, from number theory.

**Waiting on it:** `special-numbers` (`nt.special.dobinski` writes the Bell number B_5 in prose).

**Why it is open.** Analysis and number theory write B_n for the Bernoulli numbers (DLMF 24.2), and combinatorics often for the Bell numbers, which Concrete Mathematics writes ϖ_n. Physics writes B_0 for a magnetic field and B_n for the coefficients of a Fourier series.

**Recommendation.** `B_n` is the Bernoulli number, as DLMF writes it, with the conditions of decision 34: its subscript written with names the program assumes integers, never a number, never the index of a sum, so that B_0 and the coefficients `\sum_n B_n \sin nx` stay names. The Bell number is called by its name, `bell(n)`. SymPy takes B_1 = 1/2, where DLMF takes -1/2, and so does the language: the reference says so.

## 36. An index p, and an index bounded only above

**Added** on 22 November 2026, from number theory.

**Waiting on it:** `arithmetic-functions` (1), `zeta-euler-product` (1).

**Why it is open.** Number theorists let p range over the primes: `\prod_{p \mid n}` has a factor for each prime that divides n, and `\sum_{p \le x}` a term for each prime up to x. Elsewhere p is an integer like any other. And `\sum_{n \le x}` sums over the positive integers up to x, without saying where they start.

**Recommendation.** Under a sum or a product given by a condition (decision 19), an index named p ranges over the primes wherever no least value is written, as in `\prod_{p \mid n}`, `\sum_{p \le x}` and `\prod_{p \text{ prime}}`; with one, as in `\sum_{0 \le p \le n}`, it is an integer. Another index bounded only above starts at 1. Each is said in a note. The product over every prime of one of Euler's factors for ζ, as `\prod_{p \text{ prime}} \frac{1}{1 - p^{-s}}`, is its value, ζ(s), for a real part of s above 1.

## 37. A bracket of one power before a function

**Added** on 22 November 2026, from combinatorics.

**Waiting on it:** `generating-functions` (2).

**Why it is open.** Concrete Mathematics writes `[x^n] F(x)` for the coefficient of x^n in the power series of F. Read as written, the bracket is a list of one item, x^n, times F.

**Recommendation.** A bracket holding one power of a name, first in a product, is the coefficient of that power in the power series of the rest, about 0, with a note: `[x^{10}] \frac{1 - \sqrt{1 - 4x}}{2x}` is the Catalan number 16796. The power must be a whole number once it is computed; for a symbol, as in `p(n) := [x^n] \prod \ldots`, the coefficient is refused, saying so. A bracket of anything else stays a list.

## 38. A sum over an index, without limits, of binomial coefficients

**Added** on 22 November 2026, from combinatorics.

**Waiting on it:** `binomial-coefficients` (2).

**Why it is open.** Concrete Mathematics writes `\sum_k \binom{r}{k} \binom{s}{n-k}`, a sum over every integer k, its range left to the binomial coefficients, which vanish where their lower index is negative. Index notation writes `\sum_a w^a`, a sum over the values of an index of a tensor, and a sum without limits of anything else says nothing of its range.

**Recommendation.** A sum over a name with no limits, which no subscript in its body writes and which is the lower index of a binomial coefficient in it, is over every integer, with a note; its terms are summed where its binomial coefficients can be other than 0: `binomial(a, k + c)` from k = -c, `binomial(a, c - k)` up to k = c, and, of a whole number a, up to a. A sum over an index of a tensor or of data is read as before, and any other sum without limits is refused, as before. A sum of finitely many terms, from an integer to an integer, has the closed form SymPy finds for it without the conditions under which the infinite series SymPy sums it by would converge, which say nothing of it: the binomial theorem is `(x + y)^n` for every natural n.

## 39. The p of chemists

**Added** on 23 November 2026, from chemistry.

**Waiting on it:** `chem.equilibrium.henderson-hasselbalch` (1).

**Why it is open.** Chemists write pX for −log₁₀ X, as in pH, pOH and pK_a, with the p set upright: `\mathrm{p}K_\mathrm{a}`. Read as written, it was p times K_a, so that the pH of a buffer by the Henderson–Hasselbalch equation was 1.8×10⁻⁵ p + log₁₀ 2.

**Recommendation.** An upright p written against what follows it, `\mathrm{p}K_\mathrm{a}`, and `\mathrm{p}(X)`, is the cologarithm of it, −log₁₀ X, the capability `colog` of `integrations/chemistry.py`, with a note. Written apart, `\mathrm{p} \cdot K_\mathrm{a}`, it multiplies; an italic p is a letter, a pressure, a momentum or a probability; a name set upright whole, `\mathrm{pH}`, is a name of its own (decision 9); and a program that gives p a value has its own p.

## 40. Plain text and Unicode spell the constructs of LaTeX

**Added** on 23 November 2026, for agents, which type mathematics in plain text and Unicode.

**Waiting on it:** the notation matrix's plain text and Unicode forms (`asciimath`, `slash-operator`, `slash-numerator`, `text-leibniz`, `unicode-operator`, `unicode-prime`, `unicode-sign`).

**Why it is open.** Plain text and Unicode read as something else, silently, constructs that LaTeX reads: `int_0^1 x^2 dx` was the product of the names int_0, x² and dx; `log_2 8` the name log_2 times 8; `min_x x^2` a name times x²; `sin 2x` was sin(2) times x; `oo` in a script, as in `int_0^oo`, o and a factor o; and in Unicode, which is read as LaTeX is, `sin²x + cos²x` was c·o·s²·x + i·n²·s·x, the i of sin taken for the imaginary unit.

**Recommendation.** Plain text and Unicode are spellings of the constructs LaTeX writes with commands, and each reaches the syntax tree of the LaTeX it spells, or is refused:

1. The name of one of LaTeX's functions written in letters, `sin` to `csch`, `exp`, `ln`, `log`, `det`, `gcd`, `arg`, `dim`, `ker`, `sgn`, `min` and `max`, is the command wherever it applies to what follows it, as a name, a number, a bracket, a Greek letter, a command or a power does: `sin 2x` is `\sin 2x`, `sin²x` is `\sin^2 x`, `log_2 8` is `\log_2 8`, and `min_x x^2` is `\min_x x^2`. Applied to nothing, as `max` is in `max := 10`, it is a name. Either way its letters are never a product, and its i never the imaginary unit. `Re`, `Im` and `Pr` are not among them: LaTeX sets `\Re` and `\Im` as symbols, and books write the Reynolds and Prandtl numbers, and R e^{iθ}, with the same letters.
2. `int`, `iint`, `iiint` and `oint` are LaTeX's integrals before their limits or an integrand, as AsciiMath writes them: `int_0^1 x^2 dx` is `\int_0^1 x^2\,dx`. `sum`, `prod`, `lim`, `sup`, `inf`, `argmin` and `argmax` are the operators only with their scripts, `sum_{n=1}^oo` and `lim_{x->0}`: without them they are names, and `lim(...)` a call of `limit`. `sum(f, (n, a, b))`, `summation(...)` and `product(...)`, as SymPy calls them, are the sums and products `\sum_{n=a}^{b} f` and `\prod` are, unless the program defines the name; called otherwise, they are refused, saying how to write them. A script may be in round brackets, as AsciiMath groups one, `lim_(x->0)`. A subscript written without braces is one token, as TeX takes one, a number, a letter or a Greek letter's name, so that `max_iter` and `log_file` are names.
3. `oo` is infinity, one symbol in every notation, as `\infty` and ∞ are, and a whole script.
4. A solidus between differentials is Leibniz's fraction set in a line of text, as Stewart sets `dy/dx` in prose and AsciiMath types it, read as `\frac` reads it, with its guards: `d/dx f` is `\frac{d}{dx} f`, `dy/dx` is `\frac{dy}{dx}`, `d(x^3)/dx` is `\frac{d(x^3)}{dx}`, `df(x)/dx` the derivative of f(x), `d^2y/dx^2`, `∂f/∂x`, and `(del f)/(del x)`, del being AsciiMath's ∂. A differential is never called, so that `d/dx(x^3)` is the derivative of x^3, and one written with its d apart, as `dθ`, is one. After factors in a product, the fraction is a factor of its own, as `x dy/dx` is `x \frac{dy}{dx}`, and a sign before it negates it, as in `-d/dx (x^2)`. Where the program defines a name a differential is written with, as `dx := 0.01`, the solidus divides them, with a note: a user's definition wins. A solidus with no differential on one side divides, as in `dx/2` and `dr^2/(1 - 2M/r)`.
5. What still reads as a quotient or a product of names where a derivative or an integral was written is refused, with what to write: a differential nothing defines below a d nothing defines that `*` sets apart, as in `d*y/dx`, or below a name that runs d into a function the program defines, as `drate/dx` does after `rate(x) := ...`; and `int` or `integral` beside a differential at the end of a product, as in `integral x^2 dx`. Where the program defines the names, they divide and multiply, as written.
6. A function's name right after a caret is the script, whole, where it applies, as `e^ln(2)` is `e^{\ln 2}`; `int dx/x` is `\int \frac{dx}{x}`, its differential over a solidus the integrand's numerator; AsciiMath's `del` before an operand is ∂, and `mod` between two operands is `\bmod`. In Unicode, ∛ and ∜ are roots of their degree, √ takes the whole number after it, as `√16` is 4, ½ and the fraction slash, as in `3⁄4`, are fractions, ⌊ ⌋ ⌈ ⌉ are floors and ceilings, and letters, =, ( and ) set as scripts are scripts, as in `rₛ`, `eˣ` and `∑ₙ₌₁^∞`. A vulgar fraction after a number, as `2½`, is refused, since a mixed number is not the product of its parts.

The cost: in plain text `sin x y` is now sin(xy), as `\sin x y` is (decision 11), where it was sin(x) y.

## 40. A run of letters that nothing defines

**Added** on 23 November 2026, for agents, which type mathematics in plain text and Unicode.

**Waiting on it:** `implicit-relations`, where `x^3 + y^3 = 6xy` is written in plain text.

**Why it is open.** In plain text a run of letters is one name, as a program writes names, and in LaTeX and Unicode it is a product of letters, as TeX sets it (decision 10). Either reading can be the one not meant, silently: in plain text `(a+b)^2 - (a^2 + 2ab + b^2)` was 2ab − 2·ab, `xy` after `x := 2` and `y := 3` the symbol xy, and `sinx` the symbol sinx; in Unicode `sinx` was s·i·n·x, its i the imaginary unit.

**Recommendation.** Keep each notation's reading, and say so, or refuse, where the other was likely meant:

1. In plain text, a name of two or three letters, a Greek letter's name counting as one, that nothing defines, and each of whose letters the statement or the program writes alone or defines, is one name, with a note that says how to write the product: `ab` beside `(a+b)^2`, `kT` after `k := 1`. A longer word, as rate is, and a differential, as dr is, have no note: a product of four letters is written apart.
2. A name that runs one of LaTeX's functions whose argument books write after it, the trigonometric and hyperbolic functions and their inverses, exp, ln and log, into a letter, a Greek letter's name or a number, as `sinx`, `cos2theta` and `ln2` do, is in plain text one name, with a note that says how to write the function of it. In LaTeX and Unicode, where it would be a product of letters, it is refused, saying how to write either reading. The functions that begin names of their own, as arg, dim and min begin args, dims and mins, are not among them.
