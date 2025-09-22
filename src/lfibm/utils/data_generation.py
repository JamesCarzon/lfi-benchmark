from typing import Tuple
import torch


class BoxUniform:
    """Uniform distribution over a box-shaped domain."""
    
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        self.low = low
        self.high = high
        self.ndim = len(low)
        
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> torch.Tensor:
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        shape = sample_shape + (self.ndim,)
        uniform_samples = torch.rand(shape)
        return self.low + uniform_samples * (self.high - self.low)
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Check if all values are within bounds
        in_bounds = torch.all((value >= self.low) & (value <= self.high), dim=-1)
        volume = torch.prod(self.high - self.low)
        log_prob = torch.where(in_bounds, -torch.log(volume), torch.tensor(float('-inf')))
        return log_prob
