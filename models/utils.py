###############################################################
# Utility functions
###############################################################

import math
import torch

def sum_splits(values: torch.Tensor, splits: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.zeros((splits.size(0),) + values.shape[1:], dtype=values.dtype, device=values.device)
    idx = torch.repeat_interleave(torch.arange(splits.size(0), device=values.device), splits)
    out.index_add_(0, idx, values)
    return out

def mean_splits(values: torch.Tensor, splits: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    out = sum_splits(values, splits, out=out)
    # Divide by number of elements per split
    divisor = splits.view(-1, *([1]*(values.dim()-1)))
    out = out / divisor
    return out

def reduce_splits(values: torch.Tensor, splits: torch.Tensor, out: torch.Tensor | None = None, reduction: str = "sum") -> torch.Tensor:
    if reduction == "sum":
        return sum_splits(values, splits, out=out)
    elif reduction == "mean":
        return mean_splits(values, splits, out=out)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")

def cosine_cutoff(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    assert cutoff > 0.0
    return 0.5 * (torch.cos(math.pi * x / cutoff) + 1) * (x <= cutoff)

class BesselExpansion(torch.nn.Module):
    def __init__(self, size: int, cutoff: float = 5.0, trainable: bool = False) -> None:
        super().__init__()
        self.size = size
        self.register_parameter(
            "b_pi_over_c",
            torch.nn.Parameter((torch.arange(size) + 1) * math.pi / cutoff, requires_grad=trainable)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1e-10
        return torch.sin(self.b_pi_over_c * x) / x

def compute_edge_vectors_and_norms(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor | None = None,
    edge_cell: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_positions = positions[edge_index[:, 0]]
    target_positions = positions[edge_index[:, 1]]
    if edge_shift is not None:
        assert edge_cell is not None
        shift = torch.squeeze(edge_shift.unsqueeze(1) @ edge_cell, dim=1)
        target_positions = target_positions + shift
    vectors = target_positions - source_positions
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return vectors, norms

def sum_index(
    values: torch.Tensor,
    index: torch.Tensor,
    out: torch.Tensor | None = None,
    num_out: int = 0
) -> torch.Tensor:
    assert out is not None or num_out > 0
    if out is None:
        out_shape = torch.Size([num_out]) + values.shape[1:]
        out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    out.index_add_(0, index, values)
    return out
