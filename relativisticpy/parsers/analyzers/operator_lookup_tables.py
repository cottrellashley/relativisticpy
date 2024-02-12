# What are these used for?
# 1. These tables are used to simplify the logic of computing the reference typing of expressions.
# 2. They are set in LEAFT nodes from the Parser as the 'node.data_type' property.
#    The INTERNAL nodes start off as NULL in the Parser and the Semantic Analyzer computes the reuslts and injects node.data_type for these INTERNAL nodes.
# 3. Once the node.data_type surfaces up to the root node in the syntax tree, we know what the return type of the expression is.
# 4. We use the node.data_type for only two reasons:
#   
#   1 - TO FIND THE TYPE WHICH AN EXPRESSION WILL HAVE AFTER COMPUTED.
#   2 - TO STOP AND SHOW USER ERRORS BEFORE THE TREE EVEN GETS COMPUTED.


#################################################
#########  OPERATIONS LOOKUP TABLES  ############
#################################################

_undef = "undef"
_none = "none"

# Base Types
_int = "int"
_float = "float"
_string = "string"
_bool = "bool"
_symbol = "symbol" 
_tensor = "tensor"

# Compound Types
_array = "array"
_sym_expr = "sym_expr"

# Entities which will return a type
_function = "function"
_equality = "equality"


## Negative operation 'f(a) -> -a' ( NOT the minus operation 'f(a, b) -> a - b' )
negOperatorTypes = {
    _int: _int,
    _float: _float,
    _string: _undef,
    _bool: _undef, # can turn -True -> False (i.e. into not operation)
    _array: _array,
    _tensor: _tensor,
    _symbol: _symbol,
    _sym_expr: _sym_expr,
    _function: _function,
}

## Positive operation 'f(a) -> +a' ( NOT the plus operation 'f(a, b) -> a + b' )
posOperatorTypes = {
    _int: _int,
    _float: _float,
    _string: _undef,
    _bool: _undef,
    _array: _undef, # TODO: Does +[array, array, ... ] make sense ?
    _tensor: _undef,  # TODO: Does +T_{a}_{b} make sense ?
    _symbol: _symbol,
    _sym_expr: _sym_expr,
    _function: _function,
}

trigFunctionFunctionType = {
    _int: _float,
    _float: _float,
    _string: _undef,
    _bool: _undef,
    _array: _undef, # TODO: Does +[array, array, ... ] make sense ?
    _tensor: _undef,  # TODO: Does +T_{a}_{b} make sense ?
    _symbol: _sym_expr,
    _sym_expr: _sym_expr,
    _function: _sym_expr,
}

plusOperatorTypes = {
    _int: {
        _int: _int,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _float: {
        _int: _float,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _string: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _bool: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _array: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _tensor: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _tensor,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _symbol: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _sym_expr: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _function: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
}

minusOperatorTypes = {
    _int: {
        _int: _int,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _float: {
        _int: _float,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _string: {
        _int: _undef,
        _float: _undef,
        _string: _string,  # concatinate them
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _bool: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _array: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _tensor: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _tensor,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _symbol: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _sym_expr: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _function: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
}

mulOperatorTypes = {
    _int: {
        _int: _int,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _tensor,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _float: {
        _int: _float,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _tensor,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _string: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _bool: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _array: {
        _int: _array,
        _float: _array,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _array,
        _sym_expr: _array,
        _function: _array,
    },
    _tensor: {
        _int: _tensor,
        _float: _tensor,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _tensor,
        _symbol: _tensor,
        _sym_expr: _tensor,
        _function: _tensor,
    },
    _symbol: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _tensor,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _sym_expr: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _tensor,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _function: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _tensor,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
}

powOperatorTypes = {
    _int: {
        _int: _int,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _float: {
        _int: _float,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _string: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _bool: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _array: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _tensor: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _symbol: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _sym_expr: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _tensor,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _function: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _array,
        _tensor: _tensor,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
}

divOperatorTypes = {
    _int: {
        _int: _float,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _float: {
        _int: _float,
        _float: _float,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _string: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _bool: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _array: {
        _int: _array,
        _float: _array,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _array,
        _sym_expr: _array,
        _function: _array,
    },
    _tensor: {
        _int: _tensor,
        _float: _tensor,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,  # if users enter T_{x:0}
        _symbol: _tensor,
        _sym_expr: _tensor,
        _function: _tensor,
    },
    _symbol: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _sym_expr,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _sym_expr: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
    _function: {
        _int: _sym_expr,
        _float: _sym_expr,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _sym_expr,
        _sym_expr: _sym_expr,
        _function: _sym_expr,
    },
}

simplifyOperatorTypes = {
    _int: _int,
    _float: _float,
    _string: _undef,
    _bool: _undef,
    _array: _array,
    _tensor: _tensor,
    _symbol: _symbol,
    _sym_expr: _sym_expr,
    _function: _sym_expr,
}

diffOperatorTypes = {
    _int: _int, # zero
    _float: _int, # zero
    _string: _undef,
    _bool: _undef,
    _array: _array,
    _tensor: _tensor,
    _symbol: _symbol,
    _sym_expr: _sym_expr,
    _function: _sym_expr,
}

integrateOperatorTypes = {
    _int: _sym_expr,
    _float: _sym_expr,
    _string: _undef,
    _bool: _undef,
    _array: _undef,
    _tensor: _undef, # TODO : Is there a clean way to defined a Tesor integral ?
    _symbol: _symbol,
    _sym_expr: _sym_expr,
    _function: _sym_expr,
}

###################################################################################################################################################
#########   TYPES OF ASSIGNMENTS '='  #############################################################################################################
###################################################################################################################################################
#########   TYPES OF DEFINITIONS ':=' #############################################################################################################
###################################################################################################################################################


_pointer = "pointer"
_callable = "callable"

_definition_id = "definition_id"
_assignment_id = "assignment_id"
_tensor_id = "tensor_id"


assigningTypes = {
    _assignment_id : {
        _int: _pointer,
        _float: _pointer,
        _string: _pointer,
        _bool: _pointer,
        _array: _pointer,
        _tensor: _pointer,
        _symbol: _pointer,
        _sym_expr: _pointer,
        _function: _pointer,
    },
    _int: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _float: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _string: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _bool: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _array: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _tensor: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _pointer, # i.e. well defined but returns nothing
        _tensor: _pointer, # i.e. well defined but returns nothing
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _sym_expr: {
        _int: _undef,
        _float: _undef,
        _string: _undef,
        _bool: _undef,
        _array: _undef,
        _tensor: _undef,
        _symbol: _undef,
        _sym_expr: _undef,
        _function: _undef,
    },
    _function: {
        _int: _callable, # i.e. well defined but returns nothing
        _float: _callable, # i.e. well defined but returns nothing
        _string: _undef, 
        _bool: _undef,
        _array: _callable,
        _tensor: _callable,
        _symbol: _callable, # i.e. well defined but returns nothing
        _sym_expr: _callable, # i.e. well defined but returns nothing
        _function: _undef,
    },
}


ID_definitionsLookup = {
    _int: _undef,
    _float: _undef,
    _string: _undef,
    _bool: _undef,
    _array: _none,
    _tensor: _undef,
    _symbol: _none,
    _sym_expr: _undef,
    _function: _undef,
}



