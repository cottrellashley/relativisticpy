from typing import Protocol

class Symbolic(Protocol):
    """ All types which are of Symbolic Expression types. """
    pass

class Expression(Protocol):
    """ All types which are of Symbolic Expression types. """

    def substitute(self) -> 'Expression':
        ...

    def diff(self) -> 'Expression':
        ...

    def integrate(self) -> 'Expression':
        ...

class Equation(Protocol):
    """ All types which are of Symbolic Expression types. """
    
    def lhs(self) -> Expression:
        ...

    def rhs(self) -> Expression:
        ...

class SymbolArray(Protocol):
    pass