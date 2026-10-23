# AQUA Challenge - Design Document

## 1. Importance Criterion: First-Order Taylor Expansion
**Derivation:** We want to approximate the change in the loss $\Delta L$ if a weight $w$ is removed (set to 0). Using a first-order Taylor expansion around the current weight:
$\Delta L \approx L(w=0) - L(w) \approx -w \cdot \frac{\partial L}{\partial w}$
Thus, the saliency of a weight is $|w \times \nabla_w L|$.
**Why it's better than magnitude:** Magnitude pruning assumes small weights are useless, but a small weight with a massive gradient is critical to the network's dynamics. Taylor expansion balances both current magnitude and sensitivity.

## 2. The Gradient of a Masked Weight (The Correctness Trap)
**The Trap:** If a weight is masked (zeroed) during the forward pass, what happens during the backward pass and optimizer step? Many implementations zero the weight but let Adam accumulate momentum based on stale or zero-gradients. When/if the network updates, stale momentum ($m_t, v_t$) will push the zeroed weight away from zero, silently breaking sparsity or corrupting training.
**Our Solution:** In `nn/optim.py`, before applying the Adam update, we explicitly apply the mask to:
1. `p.grad *= p.mask` (Ensures zero gradient flows to zeroed weights)
2. `self.m[i] *= p.mask` (Clears momentum for pruned weights)
3. `self.v[i] *= p.mask` (Clears velocity for pruned weights)
This guarantees absolute stability.

## 3. Autodiff Bottleneck
Our current autodiff is written in Python and loops over topological nodes. The bottleneck is NumPy array creation and python object overhead per operation. To scale, we would need to trace the operations into a static C++/CUDA graph (like XLA) and execute fused kernels.

## 4. Multi-Tenant Serving at Scale
Serving a pruned model efficiently requires custom sparse kernels (e.g., Block Sparse Matrix Multiplication or CUTLASS). At thousands of RPS, we would convert our unstructured Taylor pruning into N:M structured pruning (like 2:4 sparsity supported by Nvidia Ampere architectures) to get actual hardware speedups, compiling the graph with TensorRT.
