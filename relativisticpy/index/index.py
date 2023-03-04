from relativisticpy.index.data_structure import IndexDataStructure


class Index(IndexDataStructure):
    def __init__(self, symbol, order, running, comp_type, basis, values = None):
        IndexDataStructure.__init__(self,
                            symbol      = symbol, 
                            order       = order, 
                            running     = running, 
                            basis       = basis, 
                            dimention   = int(len(basis)), 
                            values      = values if isinstance(values, int) else [i for i in range(int(len(basis)))],
                            comp_type   = comp_type if comp_type in ['contravariant','covariant'] else 'contravariant'
                            )
        self.slice = slice(0, self.dimention) if self.running else slice(self.values, self.values+1)
