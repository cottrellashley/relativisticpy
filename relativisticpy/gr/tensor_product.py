

class TensorProduct:
    # Leveraging on the work already done, create a class which can perform tensor products from just number inputs, without having to create all the Indices objects.

    def __init__(self, t1, t2, tr):
        self.t1 = t1
        self.t2 = t2
        self.tr = tr