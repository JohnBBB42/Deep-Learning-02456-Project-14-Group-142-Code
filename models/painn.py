import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss, gaussian_nll_loss
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import copy
import math
from ase.db import connect
from ase import Atoms
from torch.utils.data import DataLoader
import json
import itertools


###############################################################
# Implementations of Batch, reduce_splits, #
# cosine_cutoff, BesselExpansion, compute_edge_vectors_and_norms, and sum_index
###############################################################

"""Data object classes and related utilities."""

from collections.abc import Sequence

###############################################################
# BaseData, Data, AtomsData, GeometricData classes
###############################################################

class BaseData:
    """A dict-like base class for data objects.

    Store all tensors in a dict for easy access and enumeration.
    """

    def __init__(self, **kwargs):
        self.tensors = dict()
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __getattr__(self, key):
        # try to get from self.tensors
        if key in self.tensors:
            return self.tensors[key]
        # If not found in tensors, raise an AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value) -> None:
        # store tensors in self.tensors and everything else in self.__dict__
        if isinstance(value, torch.Tensor):
            self.tensors[key] = value
            self.__dict__.pop(key, None) 
        else:
            super().__setattr__(key, value)
            self.tensors.pop(key, None)

    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state

    def validate(self) -> bool:
        for key, tensor in self.tensors.items():
            assert isinstance(tensor, torch.Tensor), f"'{key}' is not a tensor!"
        return True

    def to(self, device: torch.device) -> None:
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}



class Data(BaseData):
    """A data object describing a homogeneous graph.

    Includes general graph information about: nodes, edges, target labels and global features.
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor = torch.tensor([]),
        edge_features: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        global_features: torch.Tensor | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.global_features = global_features
        self.targets = targets

    def validate(self) -> bool:
        super().validate()
        assert self.num_nodes > 0
        assert self.node_features.shape[0] == self.num_nodes
        assert self.node_features.ndim >= 2
        assert self.edge_index.shape[0] == self.num_edges
        assert self.edge_index.shape[0] == 0 or self.edge_index.shape[1] == 2
        assert self.edge_index.shape[0] == 0 or self.edge_index.max() < self.num_nodes
        if self.edge_features is not None:
            assert self.edge_features.shape[0] == self.num_edges
            assert self.num_edges == 0 or self.edge_features.ndim >= 2
        return True

    @property
    def num_nodes(self) -> torch.Tensor:
        # try to get num_nodes from tensors, else from node_features
        return self.tensors.get("num_nodes", self.node_features.shape[0])

    @property
    def num_edges(self) -> torch.Tensor:
        # try to get num_edges from tensors, else from edge_index
        return self.tensors.get("num_edges", self.edge_index.shape[0])

    @property
    def edge_index_source(self) -> torch.Tensor:
        return self.edge_index[:, 0]

    @property
    def edge_index_target(self) -> torch.Tensor:
        return self.edge_index[:, 1]


class AtomsData(Data):
    """A data object describing atoms as a graph with spatial information."""

    def __init__(
        self,
        node_positions: torch.Tensor,
        energy: torch.Tensor | None = None,
        forces: torch.Tensor | None = None,
        magmoms: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        volume: torch.Tensor | None = None,
        stress: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
        edge_shift: torch.Tensor | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.node_positions = node_positions
        self.energy = energy
        self.forces = forces
        self.magmoms = magmoms
        self.cell = cell
        self.volume = volume
        self.stress = stress
        self.pbc = pbc
        self.edge_shift = edge_shift

    def validate(self) -> bool:
        super().validate()
        assert self.node_positions.shape[0] == self.num_nodes
        assert self.node_positions.ndim == 2
        spatial_dim = self.node_positions.shape[1]
        if self.energy is not None:
            assert self.energy.shape == (1,)
        if self.forces is not None:
            assert self.forces.shape == (self.num_nodes, spatial_dim)
        if self.magmoms is not None:
            assert self.magmoms.shape == (self.num_nodes, 1)
        if self.cell is not None or self.pbc is not None:
            assert self.cell is not None
            assert self.pbc is not None
            assert self.cell.shape == (spatial_dim, spatial_dim)
            assert self.pbc.shape == (spatial_dim,)
        if self.volume is not None:
            assert self.cell is not None
            assert self.volume.shape == (1,)
            assert torch.isclose(self.volume, torch.linalg.det(self.cell))
        if self.stress is not None:
            assert self.cell is not None
            assert self.stress.shape == (spatial_dim, spatial_dim)
        if self.edge_shift is not None:
            assert self.edge_index is not None
            assert self.edge_shift.shape == (self.num_edges, spatial_dim)
        return True

    def any_pbc(self) -> bool:
        return self.pbc is not None and bool(torch.any(self.pbc))


class GeometricData(Data):
    """A data object describing a geometric graph with spatial information."""

    def __init__(
        self,
        node_positions: torch.Tensor,
        node_velocities: torch.Tensor | None = None,
        node_accelerations: torch.Tensor | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.node_positions = node_positions
        self.node_velocities = node_velocities
        self.node_accelerations = node_accelerations

    def validate(self) -> bool:
        super().validate()
        assert self.node_positions.shape[0] == self.num_nodes
        assert self.node_positions.ndim == 2
        spatial_dim = self.node_positions.shape[1]
        if self.node_velocities is not None:
            assert self.node_velocities.shape[0] == self.num_nodes
            assert self.node_velocities.shape[1] == spatial_dim
        if self.node_accelerations is not None:
            assert self.node_accelerations.shape[0] == self.num_nodes
            assert self.node_accelerations.shape[1] == spatial_dim
        return True

###############################################################
# Batch, collate_data and Utility functions
###############################################################

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

###############################################################
# Core PaiNN model code                                       #
###############################################################


class PaiNNInteractionBlock(nn.Module):
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.cutoff = cutoff
        self.edge_filter_net = nn.Linear(edge_size, 3 * node_size)
        self.scalar_message_net = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )
        self.U_net = nn.Linear(node_size, node_size, bias=False)
        self.V_net = nn.Linear(node_size, node_size, bias=False)
        self.a_net = nn.Sequential(
            nn.Linear(2 * node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

    def _message_function(self, node_states_scalar, node_states_vector, edge_states, edge_vectors, edge_norms, edge_index):
        filter_weight = self.edge_filter_net(edge_states)
        filter_weight = filter_weight * cosine_cutoff(edge_norms, self.cutoff)

        scalar_output = self.scalar_message_net(node_states_scalar)
        src_nodes = edge_index[:,0]
        dst_nodes = edge_index[:,1]

        filter_output = filter_weight * scalar_output[src_nodes]

        gate_nodes, gate_edges, messages_scalar = torch.split(filter_output, self.node_size, dim=1)
        gate_nodes = gate_nodes.unsqueeze(1)
        gate_edges = gate_edges.unsqueeze(1)

        gated_node_states_vector = node_states_vector[src_nodes]*gate_nodes
        gated_edge_vectors = gate_edges * edge_vectors.unsqueeze(2)
        messages_vector = gated_node_states_vector + gated_edge_vectors

        delta_node_states_scalar_m = sum_index(messages_scalar, dst_nodes, torch.zeros_like(node_states_scalar))
        delta_node_states_vector_m = sum_index(messages_vector, dst_nodes, torch.zeros_like(node_states_vector))

        return delta_node_states_scalar_m, delta_node_states_vector_m

    def _node_state_update_function(self, node_states_scalar, node_states_vector):
        Uv = self.U_net(node_states_vector)
        Vv = self.V_net(node_states_vector)
        Vv_square_norm = torch.sum(Vv**2, dim=1)

        a = self.a_net(torch.cat((node_states_scalar, Vv_square_norm), dim=1))
        a_ss, a_sv, a_vv = torch.split(a, self.node_size, dim=1)

        inner_prod_Uv_Vv = torch.sum(Uv*Vv, dim=1)
        delta_node_states_scalar_u = a_ss + a_sv*inner_prod_Uv_Vv
        delta_node_states_vector_u = a_vv.unsqueeze(1)*Uv
        return delta_node_states_scalar_u, delta_node_states_vector_u

    def forward(self, node_states_scalar, node_states_vector, edge_states, edge_vectors, edge_norms, edge_index):
        delta_node_states_scalar_m, delta_node_states_vector_m = self._message_function(
            node_states_scalar, node_states_vector, edge_states, edge_vectors, edge_norms, edge_index
        )

        node_states_scalar = node_states_scalar + delta_node_states_scalar_m
        node_states_vector = node_states_vector + delta_node_states_vector_m

        delta_node_states_scalar_u, delta_node_states_vector_u = self._node_state_update_function(
            node_states_scalar, node_states_vector
        )

        node_states_scalar = node_states_scalar + delta_node_states_scalar_u
        node_states_vector = node_states_vector + delta_node_states_vector_u
        return node_states_scalar, node_states_vector

class PaiNN(nn.Module):
    def __init__(self, node_size=64, edge_size=20, num_interaction_blocks=3, cutoff=5.0,
                 pbc=False, use_readout=True, num_readout_layers=2, readout_size=1,
                 readout_reduction="sum"):
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.num_interaction_blocks = num_interaction_blocks
        self.cutoff = cutoff
        self.pbc = pbc
        self.use_readout = use_readout
        self.num_readout_layers = num_readout_layers
        self.readout_size = readout_size
        self.readout_reduction = readout_reduction

        num_embeddings = 119
        self.node_embedding = nn.Embedding(num_embeddings, node_size)
        self.edge_expansion = BesselExpansion(edge_size, cutoff)
        self.interaction_blocks = nn.ModuleList(
            PaiNNInteractionBlock(node_size, edge_size, cutoff)
            for _ in range(num_interaction_blocks))

        if self.use_readout:
            layers = []
            for _ in range(num_readout_layers - 1):
                layers.append(nn.Linear(node_size, node_size))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(node_size, readout_size))
            self.readout_net = nn.Sequential(*layers)
        else:
            self.readout_net = None

    def forward(self, batch):
        node_states_scalar = self.node_embedding(batch.node_features.squeeze(-1))
        node_states_vector = torch.zeros(
            node_states_scalar.size(0), 3, self.node_size,
            dtype=node_states_scalar.dtype,
            device=node_states_scalar.device
        )

        # If we have multiple graphs batched, cell is (num_graphs, 3, 3).
        # We need a (num_edges, 3, 3) cell_for_edges for each edge.
        if (hasattr(batch, 'cell') and batch.cell is not None and
            hasattr(batch, 'edge_shift') and batch.edge_shift is not None):
            # Create a per-edge cell tensor
            # batch.edge_data_index maps each edge to its graph index
            cell_for_edges = batch.cell[batch.edge_data_index]  # shape: (total_edges, 3, 3)

            edge_vectors, edge_norms = compute_edge_vectors_and_norms(
                batch.node_positions, batch.edge_index,
                batch.edge_shift, cell_for_edges
            )
        else:
            # If we don't have multiple graphs or no cell/edge_shift, just pass them directly.
            edge_vectors, edge_norms = compute_edge_vectors_and_norms(
                batch.node_positions, batch.edge_index,
                getattr(batch, 'edge_shift', None),
                getattr(batch, 'cell', None)
            )


        edge_vectors = edge_vectors / (edge_norms+1e-10)
        edge_states = self.edge_expansion(edge_norms)


        for block in self.interaction_blocks:
            node_states_scalar, node_states_vector = block(
                node_states_scalar, node_states_vector,
                edge_states, edge_vectors, edge_norms, batch.edge_index
            )


        if self.use_readout:
            node_states_scalar = self.readout_net(node_states_scalar)

        if self.readout_reduction:
            output_scalar = reduce_splits(node_states_scalar, batch.num_nodes, reduction=self.readout_reduction)
        else:
            output_scalar = node_states_scalar

        return output_scalar
        node_states_scalar = node_states_scalar + delta_node_states_scalar_u
        node_states_vector = node_states_vector + delta_node_states_vector_u
        # Return updated node states
        return node_states_scalar, node_states_vector
