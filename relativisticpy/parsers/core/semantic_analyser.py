######################  WHAT DOES THE SEMANTIC ANALYZER DO (FOR RELATIVISTICPY SPECIFICALLY)
# We know Sematic Analyzers are complex and is there to check whether the AST form a sensible set of instructions in the programming language. 
# But we are being quite flexible with the word in this package and adding some of the meaning of what a Semantic Analyzer does:
#
#
#    Since this package is built without it's own symbolic/numeric/plotting tools/engines
#    We get into the issue where user wants this layer abstracted out - they simpley want to write equations and see the result.
#   
#    We run into the issue where different tools we have different ways of interacting with each other:
#    Here is a list of package current/future interfaces we currently/will have:
#
#   SymPy <-> Numpy/Scipy
#   Scipy <-> MatplotLib
#   Numpy <-> MatplotLib
#   SymPy <-> MatplotLib
#   SymPy <-> RelativisticPy
#   Numpy/Scipy <-> RelativisticPy
#   MatplotLib <-> RelativisticPy
#   
#   A generic Abstract Syntax Tree does not know of this layer of context. It simply parses the tokens into a Tree.
#   
#   It is the job of the Semantic Analyzer (SA) to:
#       1. Check the AST and see if it makes grammatical sense. Throwing an inforemative error if it does not.
#       2. Add a layer of context to the AST so that when we implement the executor of the tree, it needs only to worry about implementation.
#       3. Add context as to what type the user wants from the AST as a whole. Does he AST return a graph? equation? tensor? if we know we can help optimize.
#
#   Since TECHNICALLY this whole parser module should NOT know the implementation of the methods and even less what tools (packages) are being used to implememt
#   the computations, we shall keep the langauge and naming abstract in such a way to keep the parser at a lower layer:
#   Language:
#   
#       Types: Tensor - Symbolic Expression - Numerical Expression / Object - Visual Object
#               



class SemanticAnalyzer:
    
    def __init__(self, AST):
        self.AST = AST

    