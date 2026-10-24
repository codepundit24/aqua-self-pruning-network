import numpy as np
import matplotlib.pyplot as plt
from engine.tensor import Tensor
from engine.ops import softmax_cross_entropy
from nn.layers import Linear
from nn.optim import Adam
from prune.core import Pruner

# Dataset generation
def get_data(samples=1000):
    N = samples // 2
    X = np.zeros((samples, 2))
    y = np.zeros((samples, 2))
    for j in range(2):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix, j] = 1
    return X, y

sparsities = [0.0, 0.5, 0.75, 0.90, 0.95]
results_acc = []
results_actual_sparsity = []

X, y = get_data()

for target in sparsities:
    print(f"Training for target sparsity: {target*100}%...")
    model = [Linear(2, 64), Linear(64, 64), Linear(64, 2)]
    
    def forward(x):
        h1 = model[0](x).relu()
        h2 = model[1](h1).relu()
        return model[2](h2)
    
    params = []
    for layer in model: params.extend(layer.parameters())
    
    optimizer = Adam(params, lr=0.01)
    pruner = Pruner(params, target_sparsity=target, schedule_steps=200)
    
    for epoch in range(250):
        inputs = Tensor(X)
        logits = forward(inputs)
        loss = softmax_cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pruner.step()
        
    # Calculate accuracy
    final_logits = forward(Tensor(X)).data
    preds = np.argmax(final_logits, axis=1)
    true_labels = np.argmax(y, axis=1)
    acc = np.mean(preds == true_labels)
    
    # Calculate actual sparsity
    actual_sparsity = sum(np.sum(p.mask == 0) for p in pruner.params) / sum(p.data.size for p in pruner.params)
    
    results_acc.append(acc * 100)
    results_actual_sparsity.append(actual_sparsity * 100)
    print(f"Result -> Sparsity: {actual_sparsity*100:.2f}%, Accuracy: {acc*100:.2f}%\n")

# Save raw numbers
with open("benchmarks/pareto_results.txt", "w") as f:
    f.write("Target_Sparsity(%),Actual_Sparsity(%),Accuracy(%)\n")
    for t, s, a in zip(sparsities, results_actual_sparsity, results_acc):
        f.write(f"{t*100},{s:.2f},{a:.2f}\n")

# Plot and save
plt.figure(figsize=(8, 5))
plt.plot(results_actual_sparsity, results_acc, marker='o', linestyle='-', color='b')
plt.title("Sparsity vs Accuracy (Pareto Curve)")
plt.xlabel("Actual Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("benchmarks/pareto_curve.png")
print("Saved pareto_results.txt and pareto_curve.png in benchmarks folder.")