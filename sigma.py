import math

class Sigmoid:
    def __init__(self):
        pass
    
    def compute(self, v):
        return 1/(1+math.exp(-v))

    def diff(self, v):
        return self.compute(v)*(1-self.compute(v))
