from enum import Enum


class WorkbookConstants(Enum):
    # Standard Defs
    COORDINATES = "Coordinates"

    # Tensor/Object Types
    METRIC = "Metric"
    RICCI = "Ricci"
    RIEMANN = "Riemann"
    COVDERIVATIVE = "CovariantDerivative"
    DERIVATIVE = "Derivative"

    # Symbol Setters
    METRICSYMBOL = "MetricSymbol"
    RICCISYMBOL = "RicciSymbol"
    RIEMANNSYMBOL = "RiemannSymbol"
    COVDERIVATIVESYMBOL = "CovariantDerivativeSymbol"
    DERIVATIVESYMBOL = "DerivativeSymbol"
