import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, parents=None, op_name=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self.parents = parents or []
        self.op_name = op_name
        self.mask = None # For pruning

    def apply_mask(self):
        if self.mask is not None:
            self.data *= self.mask
            if self.grad is not None:
                self.grad *= self.mask

    def backward(self):
        if not self.requires_grad:
            return
        
        self.grad = np.ones_like(self.data)
        
        # Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)
        
        for v in reversed(topo):
            if v.mask is not None:
                v.apply_mask()
            v._backward()
            if v.mask is not None:
                v.apply_mask()

    # Operations
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], op_name='+')
        def _backward():
            if self.requires_grad: self.grad += __class__.handle_broadcast(out.grad, self.data.shape)
            if other.requires_grad: other.grad += __class__.handle_broadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], op_name='-')
        def _backward():
            if self.requires_grad: self.grad += __class__.handle_broadcast(out.grad, self.data.shape)
            if other.requires_grad: other.grad += __class__.handle_broadcast(-out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], op_name='*')
        def _backward():
            if self.requires_grad: self.grad += __class__.handle_broadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad: other.grad += __class__.handle_broadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out
        
    def matmul(self, other):
        out = Tensor(self.data @ other.data, requires_grad=(self.requires_grad or other.requires_grad), parents=[self, other], op_name='@')
        def _backward():
            if self.requires_grad: self.grad += out.grad @ other.data.T
            if other.requires_grad: other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, parents=[self], op_name='relu')
        def _backward():
            if self.requires_grad: self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        return out

    @staticmethod
    def handle_broadcast(grad, target_shape):
        if grad.shape == target_shape: return grad
        ndims_added = len(grad.shape) - len(target_shape)
        for _ in range(ndims_added): grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1: grad = grad.sum(axis=i, keepdims=True)
        return grad
