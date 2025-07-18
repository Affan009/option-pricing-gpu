import torch
import option_pricer
import time
import math

OptionType = option_pricer.OptionType
BarrierType = option_pricer.BarrierType
OptionStyle = option_pricer.OptionStyle

def price_option(option_type, barrier_type, style, S, K, r, sigma, T, barrier=0.0, steps=100, paths=100_000, seed=None):
    if seed is None:
        seed = int(time.time())
    results = torch.zeros(paths, dtype=torch.float32, device="cuda")
    option_pricer.runMonteCarlo(
        results, paths, S, K, r, sigma, T,
        barrier, steps, option_type, barrier_type, style, seed
    )
    mean = results.mean().item()
    stderr = results.std().item() / math.sqrt(paths)
    return mean, stderr
