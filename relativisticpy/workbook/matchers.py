# I defined a callback method which will match on a string and return true or false. in this case I choose a tensor
import re

def match_tensors(i):
    """
    Will match on strings of form:

            WORD_{LETTER:INTEGER}_{LETTER:INTEGER}_.........REPEAT
    """

    string = i
    rank = string.count("_") + string.count("^")
    if rank > 0:
        pattern = (
            lambda x: "([a-zA-Z]+)([_^]\{[a-zA-Z]+\}|[_^]\{[a-zA-Z]+\=[0-9]}){"
            + str(x)
            + "}(?=(\*|\)|\+|\-|\/|$))"
        )
        pattern2 = (
            lambda x: "([a-zA-Z]+)([_^]\{[a-zA-Z]+\}|[_^]\{[a-zA-Z]+\:[0-9]}){"
            + str(x)
            + "}(?=(\*|\)|\+|\-|\/|$))"
        )
        Total = re.match(pattern(rank), string)
        Total2 = re.match(pattern2(rank), string)
        return bool(Total) or bool(Total2)
    else:
        return False

def match_symbol(i):
    return bool(re.match(r"^\w+$", i))
    

# Notice the node type is object, but when we enter these strings, it will call the matcher method and change the node key to "TENSOR"
variable_matchers = [
    {"node": "object", "node_key": "tensor", "string_matcher_callback": match_tensors}
]
