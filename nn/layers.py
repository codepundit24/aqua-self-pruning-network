import numpy as np
from engine.tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features):
        # He initialization
        bound = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features) * bound, requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
        
    def __call__(self, x):
        # Apply mask before forward pass to ensure strict sparsity
        self.weight.apply_mask()
        return x.matmul(self.weight) + self.bias
        
    def parameters(self):
        return [self.weight, self.bias]
