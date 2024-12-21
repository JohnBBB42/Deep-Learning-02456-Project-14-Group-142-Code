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

