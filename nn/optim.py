import numpy as np

class Adam:
    def __init__(self, parameters, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None: continue
            
            # THE TRAP FIX: Ensure masked gradients are absolutely zero
            # AND zero out moments for pruned weights to prevent stale state corruption.
            if p.mask is not None:
                p.grad *= p.mask
                self.m[i] *= p.mask
                self.v[i] *= p.mask

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad ** 2)

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= update
            
            # Enforce mask physically on data after step
            if p.mask is not None:
                p.data *= p.mask
