import re
from relativisticpy.deserializers.mathify import Mathify


def indices_from_string(index_cls, indices_cls, string_arg):
    # Representation 1 (r1): String Representation of Indices / Idx objects => deserialises string into Indices(Idx's, ...)
    def _isr1(arg: str):
        return (
            bool(
                re.search(
                    "^((\^|\_)(\{)(\}))+$",
                    re.sub("[^\^^\_^\{^\}]", "", arg).replace(" ", ""),
                )
            )
            if isinstance(arg, str)
            else False
        )

    def _r1running(arg: str):
        return not bool(re.search("^[^=]*(\:)([0-9]+)[^=]*$", arg))

    def _r1symbol(arg: str):
        return (
            re.search(r"[a-zA-Z]+", arg).group()
            if re.search(r"[a-zA-Z]+", arg)
            else None
        )

    def _r1values(arg: str):
        return re.search(r"[0-9]+", arg).group() if re.search(r"[0-9]+", arg) else None

    def _r1covariant(arg: str):
        return (
            arg[0] == "_" if isinstance(arg, str) else True
        )  # Always default to coveriant

    def _r1split(arg: str):
        return (
            [item for item in re.split("(?=[_^])", arg) if item] if _isr1(arg) else arg
        )

    def r1_deserialiser(arg: str):
        return (
            indices_cls(
                *[
                    index_cls(
                        symbol=_r1symbol(i),
                        covariant=_r1covariant(i),
                        values=int(_r1values(i)) if not _r1running(i) else None,
                    )
                    for i in _r1split(arg)
                ]
            )
            if _isr1(arg)
            else None
        )

    # Any future/other string representations of indices goes bellow as r2, r3, etc...

    # Once we have more representations, we can add a representation matcher, which mathes the rn type and passes responsibility to relevant parser.

    indices = r1_deserialiser(string_arg)
    return indices
