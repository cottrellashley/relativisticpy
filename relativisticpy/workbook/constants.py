from enum import Enum


class WorkbookConstants(Enum):
    # Standard Defs
    COORDINATES = "Coordinates"

    # Tensor/Object Types
    METRIC = "Metric"
    METRICSCALAR = "MetricScalar"
    RICCI = "Ricci"
    RIEMANN = "Riemann"
    COVDERIVATIVE = "CovariantDerivative"
    DERIVATIVE = "Derivative"
    CONNECTION = 'Connection'
    EINSTEINTENSOR = "EinsteinTensor"

    # Symbol Setters
    METRICSYMBOL = "MetricSymbol"
    EINSTEINTENSORSYMBOL = "EinsteinTensorSymbol"
    RICCISYMBOL = "RicciSymbol"
    RIEMANNSYMBOL = "RiemannSymbol"
    COVDERIVATIVESYMBOL = "CovariantDerivativeSymbol"
    DERIVATIVESYMBOL = "DerivativeSymbol"
