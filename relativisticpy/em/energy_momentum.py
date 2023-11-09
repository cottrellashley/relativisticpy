from relativisticpy.core import MultiIndexObject, Indices, Metric
from typing import Union
from abc import ABC, abstractmethod
from relativisticpy.core.exceptions import ArgumentException
from relativisticpy.gr.connection import Connection
from relativisticpy.utils import SymbolArray

class PhysicalTensor(MultiIndexObject):
    pass