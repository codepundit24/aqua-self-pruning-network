import numpy as np
from .tensor import Tensor

def softmax_cross_entropy(logits, targets):
    # Numerically stable softmax
    max_logits = np.max(logits.data, axis=1, keepdims=True)
    exp_logits = np.exp(logits.data - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    batch_size = logits.data.shape[0]
    # targets is expected to be one-hot
    loss_val = -np.sum(targets * np.log(probs + 1e-12)) / batch_size
    
    out = Tensor(loss_val, requires_grad=logits.requires_grad, parents=[logits], op_name='softmax_ce')
    def _backward():
        if logits.requires_grad:
            logits.grad += (probs - targets) / batch_size
    out._backward = _backward
    return out
