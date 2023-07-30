import re
from relativisticpy.providers import Sympify


def deserialisable_tensor(cls):
    """
    Class decorator. Injects descerilization of a tensor object. The specific __init__ rules are determined by class taking the attribute.

    Required class property definitions:
            >>> ClassName._cls_idx
            >>> ClassName._cls_idcs

    Example:

            >>> @deserialisable_tensor
            >>> class Tensor:
            >>>     _cls_idx = Idx  # <=== Must provide the Index class which will be instantiated by from_string()
            >>>     _cls_idcs = Indices  # <=== Must provide the Indices class which will be instantiated by from_string()
            >>>     
            >>>     def __init__(self, indices, components, basis):  # <==== Tensor object taking this decorator must have 
            >>>         ...
            >>>
            >>> Decorator injects the implementation of from_string() function in class which can be used:
            >>> Tensor.from_string(indices = '_{a}_{b}', components = '[[1, 0],[0, r**2]]', basis = '[r, theta]')
    """

    # Representation 1 (r1): String Representation of Indices / Idx objects => deserialises string into Indices(Idx's, ...)
    def _isr1(arg: str): return bool(re.search("^((\^|\_)(\{)(\}))+$", re.sub('[^\^^\_^\{^\}]',"", arg).replace(" ",''))) if isinstance(arg, str) else False
    def _r1running(arg: str): return not bool(re.search('^[^=]*(\:)([0-9]+)[^=]*$', arg))
    def _r1symbol(arg: str): return re.search(r'[a-zA-Z]+', arg).group() if re.search(r'[a-zA-Z]+', arg) else None
    def _r1values(arg: str): return re.search(r'[0-9]+', arg).group() if re.search(r'[0-9]+', arg) else None
    def _r1covariant(arg: str): return arg[0] == '_' if isinstance(arg, str) else True # Always default to coveriant
    def _r1split(arg: str): return [item for item in re.split('(?=[_^])', arg) if item] if _isr1(arg) else arg
    def r1_deserialiser(arg: str): return cls._cls_idcs(*[cls._cls_idx(symbol = _r1symbol(i), covariant=_r1covariant(i), values=int(_r1values(i)) if not _r1running(i) else None) for i in _r1split(arg)]) if _isr1(arg) else None

    # Any future/other string representations of indices goes bellow as r2, r3, etc...

    # Once we have more representations, we can add a representation matcher, which mathes the rn type and passes responsibility to relevant parser.

    def deserialiser(indices, components, basis):
        return cls(indices = r1_deserialiser(indices), components = Sympify(components), basis = Sympify(basis))

    cls.from_string = deserialiser
    return cls
