###############################################################
# Batch, collate_data and Utility functions
###############################################################

import math
import torch

class Batch(Data):
    """An object representing a batch of data.

    Typically a disjoint union of graphs.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._node_data_index: torch.Tensor | None = None
        self._edge_data_index: torch.Tensor | None = None

    def validate(self) -> bool:
        for key, tensor in self.tensors.items():
            assert isinstance(tensor, torch.Tensor), f"'{key}' is not a tensor!"
        assert self.node_features.shape[0] == torch.sum(self.num_nodes)
        assert self.node_features.shape[0] == 0 or self.node_features.ndim >= 2
        assert self.edge_index.shape[0] == torch.sum(self.num_edges)
        assert self.edge_index.shape[0] == 0 or self.edge_index.shape[1] == 2
        assert self.edge_index.shape[0] == 0 or self.edge_index.max() < self.node_features.shape[0]
        assert self.num_data >= 1
        if self.edge_features is not None:
            assert self.edge_features.shape[0] == self.edge_index.shape[0]
            assert self.edge_features.ndim >= 2
        if self.global_features is not None:
            assert self.global_features.shape[0] == self.num_data
        if self.targets is not None:
            assert self.targets.ndim >= 2
        return True

    @property
    def num_data(self) -> int:
        # Number of graphs in the batch is the length of the num_nodes tensor.
        return self.num_nodes.shape[0]

    @property
    def node_data_index(self) -> torch.Tensor:
        if self._node_data_index is None:
            self._node_data_index = torch.repeat_interleave(
                torch.arange(self.num_nodes.shape[0], device=self.num_nodes.device), self.num_nodes)
        return self._node_data_index

    @property
    def edge_data_index(self) -> torch.Tensor:
        if self._edge_data_index is None:
            self._edge_data_index = torch.repeat_interleave(
                torch.arange(self.num_edges.shape[0], device=self.num_edges.device), self.num_edges)
        return self._edge_data_index


def collate_data(list_of_data: Sequence[Data]) -> Batch:
    """Collate a list of data objects into a batch object.

    The input graphs are combined into a single graph as a disjoint union by
    concatenation of all data and appropriate adjustment of the edge_index.
    """
    batch = dict()
    batch["num_nodes"] = torch.tensor([d.num_nodes for d in list_of_data])
    batch["num_edges"] = torch.tensor([d.num_edges for d in list_of_data])
    offset = torch.cumsum(batch["num_nodes"], dim=0) - batch["num_nodes"]
    batch["edge_index"] = torch.cat([d.edge_index + offset[i] for i, d in enumerate(list_of_data)])
    for k in list_of_data[0].tensors.keys():
        if k not in batch.keys():
            try:
                if k == "cell" or k == "stress":
                    batch[k] = torch.cat([d.tensors[k].unsqueeze(0) for d in list_of_data])
                else:
                    batch[k] = torch.cat([torch.atleast_2d(d.tensors[k]) for d in list_of_data])
            except Exception as e:
                raise Exception(f"Failed to add '{k}' to batch:", e)
    return Batch(**batch)

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
