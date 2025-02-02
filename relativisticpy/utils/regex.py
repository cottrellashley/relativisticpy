import re
from typing import Union

def tensor_key_patern() -> re.Pattern:
    return re.compile("([a-zA-Z]+)")


def _r1_bare_tensor() -> re.Pattern:
    return re.compile("^((\^|\_)(\{)(\}))+$")


def get_tensor_key(tensor_str: str) -> Union[str, None]:
    return (
        re.findall(tensor_key_patern(), tensor_str)[0]
        if bool(re.search(tensor_key_patern(), tensor_str))
        else None
    )


def str_is_r1_tensor(tensor_str: str) -> bool:
    return bool(
        re.search(
            "^((\^|\_)(\{)(\}))+$",
            re.sub("[^\^^\_^\{^\}]", "", tensor_str.replace(" ", "")),
        )
    )


def str_has_symbol(def_key: str) -> bool:
    bool(re.match(r"\b\w+Symbol\b", def_key))


def str_is_tensors(i : str) -> bool:
    """
    Will match on strings of form:

            WORD_{LETTER:INTEGER}_{LETTER:INTEGER}_.........REPEAT
    """
    string = i
    rank = string.count("_") + string.count("^")
    if rank > 0:
        pattern2 = (
            lambda x: "([a-zA-Z]+)([_^]\{[a-zA-Z]+\}|[_^]\{[a-zA-Z]+\:[0-9]}){"
            + str(x)
            + "}(?=(\*|\)|\+|\-|\/|$))"
        )
        Total2 = re.match(pattern2(rank), string)
        return bool(Total2)
    else:
        return False


def is_symbol_object_key(string: str) -> bool:
    return bool(re.match(r"\b\w+Symbol\b", string))

def is_symbol(string: str) -> bool:
    return bool(re.match(r"^\w+$", string))

def extract_tensor_symbol(string: str) -> Union[str, None]:
    return (
        re.findall("([a-zA-Z]+)", string)[0]
        if bool(re.search("([a-zA-Z]+)", string))
        else None
    )


def extract_tensor_indices(string: str) -> Union[str, None]:
    symbol = (
        extract_tensor_symbol(string) if extract_tensor_symbol(string) != None else ""
    )
    return string.replace(symbol, "")


def tensor_index_running(string: str) -> bool:
    return not bool(re.search(r"\d", string))
