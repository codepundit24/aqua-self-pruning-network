import numpy as np

def calculate_taylor_saliency(tensor):
    """
    First-order Taylor expansion criterion.
    Added a tiny magnitude fallback (1e-9 * abs(weight)) to mathematically break ties 
    when gradients are exactly zero, preventing mass network collapse.
    """
    if tensor.grad is None:
        return np.abs(tensor.data)
    return np.abs(tensor.data * tensor.grad) + 1e-9 * np.abs(tensor.data)

class Pruner:
    def __init__(self, model_params, target_sparsity=0.8, schedule_steps=200):
        self.params = [p for p in model_params if len(p.data.shape) > 1]
        for p in self.params:
            if p.mask is None:
                p.mask = np.ones_like(p.data)
        self.target_sparsity = target_sparsity
        self.schedule_steps = schedule_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step > self.schedule_steps:
            return

        # Cubic sparsity schedule
        progress = self.current_step / self.schedule_steps
        current_sparsity = self.target_sparsity * (1 - (1 - progress)**3)

        # Gather all scores across the entire network
        all_scores = []
        for p in self.params:
            score = calculate_taylor_saliency(p)
            all_scores.extend(score.flatten())

        if not all_scores: return

        # Find threshold based on the exact percentage of total weights
        k = int(len(all_scores) * current_sparsity)
        if k == 0: return
        threshold = np.partition(all_scores, k)[k]

        # Apply mask
        for p in self.params:
            score = calculate_taylor_saliency(p)
            p.mask = (score >= threshold).astype(np.float32)
            p.apply_mask()