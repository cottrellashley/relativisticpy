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
4. after a parenthesised expression it is a Pochhammer length;
5. otherwise it is part of the name, whatever the base: Greek letters, numbers, signs, ⊙. A built-in tensor claims a subscripted name only while a metric is in force;
6. `f_x` is a partial derivative only on a defined function of x, with nothing defining `f_x`;
7. in a letter run the subscript binds to the last letter, as in TeX (`iE_1` is i·E_1);
8. a comma separates labels on a name that is not a tensor.

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

## 7. What a prime means

**Waiting on it:** `primed-names` (11).

**Why it is open.** It can be a derivative or another frame (x', t', E', F'^{μν}, k'^μ, μ'). Today a prime is always a derivative, so `x' :=` is refused and `\mathbf{r}(t') :=` silently becomes an equation.

**Recommendation.** A prime is a derivative only on a function of one variable: one the program defined, or an unknown function written with its argument. Elsewhere, and always on the left of `:=`, it is part of the name. On an index it marks another frame's index.

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

## 11. How far an unbracketed function argument runs

**Waiting on it:** `function-argument-extent` (2).

**Why it is open.** Today `\sin (2k-1) x` is x·sin(2k − 1), and `\sin\theta \frac{\partial Y}{\partial\theta}` is sin(θ·∂Y/∂θ).

**Recommendation.** Follow the books: the argument is the following product of factors, up to an operator, a fraction, a derivative or another function. So `\sin (2k-1)x` is sin((2k−1)x) and `\sin\theta\,\frac{a}{b}` is sin θ·a/b. Give a note where both readings are plausible.

## 12. The identity, and numbers beside matrices

**Waiting on it:** `identity-matrix` (7).

**Why it is open.** Books write `A - \lambda I`, `\mathbf{1}`, `\mathbb{1}`, and the Dirac `(\gamma^0 E - \gamma^3 p - m)`. Today `I` is a plain symbol (the imaginary unit is `i`), and a number added to a matrix is refused.

**Recommendation.** An undefined `I`, `\mathbf{1}`, `\mathbb{1}` or 𝟙 in matrix arithmetic is the identity of the matching size, with a note; a defined `I` keeps its value. A number added to a square matrix inside a matrix equation is that multiple of the identity, with a note.

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

## 15. Book names that collide with the language's or with each other

**Waiting on it:** `special-functions` (18), `step-delta` (6), `normal-cdf` (2).

**Why it is open.** N(x) is the normal distribution function in Hull and numerical evaluation in the language; Black–Scholes silently gives 8.42 instead of 4.76. Φ is the normal distribution function or a potential. Γ is Euler's gamma or a Christoffel symbol, and H_n the physicists' or the probabilists' Hermite polynomial. u_c(t) is a step or a velocity, δ_0 a phase shift or the Kronecker delta, π_j a vector or the constant.

**Recommendation.** A user's definition always wins. Otherwise:

1. `\Gamma(x)` with one bracketed argument and no indices is Euler's gamma; with indices it stays Christoffel;
2. `\mathrm{B}(a, b)` is the Beta function, and `H_n` the physicists' Hermite polynomial (probabilists write He_n);
3. `u_c(t)` is the unit step where u is not defined;
4. `N(x)` and `\Phi(x)` are the standard normal distribution function, with a note. This retires `N` as numerical evaluation in LaTeX, where `evalf` and `numerical` remain;
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

