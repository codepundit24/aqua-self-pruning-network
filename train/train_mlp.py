import numpy as np
from engine.tensor import Tensor
from engine.ops import softmax_cross_entropy
from nn.layers import Linear
from nn.optim import Adam
from prune.core import Pruner

# Synthetic Spiral Dataset
def get_data(samples=1000):
    N = samples // 2
    X = np.zeros((samples, 2))
    y = np.zeros((samples, 2))
    for j in range(2):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix, j] = 1 # one-hot
    return X, y

model = [Linear(2, 64), Linear(64, 64), Linear(64, 2)]
def forward(x):
    h1 = model[0](x).relu()
    h2 = model[1](h1).relu()
    return model[2](h2)

params = []
for layer in model: params.extend(layer.parameters())

optimizer = Adam(params, lr=0.01)
pruner = Pruner(params, target_sparsity=0.8, schedule_steps=200)

X, y = get_data()

for epoch in range(250):
    inputs = Tensor(X)
    logits = forward(inputs)
    loss = softmax_cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    pruner.step()
    
    if epoch % 50 == 0:
        sparsity = sum(np.sum(p.mask == 0) for p in pruner.params) / sum(p.data.size for p in pruner.params)
        print(f"Epoch {epoch} | Loss: {loss.data:.4f} | Sparsity: {sparsity*100:.2f}%")
