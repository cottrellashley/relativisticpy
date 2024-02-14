# Sympy Dependencies: By importing all the sympy functionality via this file, we can controll what it used and can also swap out implementations in future.
from sympy import (
    Symbol,
    Basic,
    Rational,
    Function,
    Interval,
    Pow,
    Order,
    Sum,
    Product,
    O,
    LaplaceTransform,
    diff,
    integrate,
    simplify,
    tensorproduct,
    symbols,
    residue,
    laplace_transform,
    inverse_laplace_transform,
    inverse_mellin_transform,
    mellin_transform,
    fourier_transform,
    inverse_fourier_transform,
    sine_transform,
    inverse_sine_transform,
    cosine_transform,
    inverse_cosine_transform,
    hankel_transform, 
    inverse_hankel_transform,
    zeros,
    permutedims,
    solve,
    dsolve,
    expand,
    latex,

    # Series Source: https://docs.sympy.org/latest/modules/series/series.html
    limit,
    Limit,
    fourier_series,
    sequence,
    series,
    SeqFormula,

    # Euler Source: https://docs.sympy.org/latest/modules/calculus/index.html
    euler_equations,
    is_decreasing,
    is_increasing,
    is_monotonic,

    # Functions Source: https://docs.sympy.org/latest/modules/functions/elementary.html
    exp_polar,
    bell,
    bernoulli,
    binomial,
    gamma,
    conjugate,
    hyper,
    catalan,
    euler,
    factorial,
    fibonacci,
    harmonic,
    oo,
    log,
    I,
    E,
    N,
    pi,
    exp,
    sin,
    asin,
    sinh,
    asinh,
    cos,
    acos,
    cosh,
    acosh,
    tan,
    atan,
    tanh,
    atanh,
    DiracDelta,
    Heaviside,

    # Algebras Source: https://docs.sympy.org/latest/modules/algebras.html
    Quaternion,

    # Discrete stuff source: 
    fft,  # Fast Fourier Trnasform
    ifft
)
from sympy import MutableDenseNDimArray as SymbolArray
from .sympy import root

# Implement `function` - `constant` - `infinity`
