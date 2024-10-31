"""Polarizable atom interaction neural network (PaiNN) model (https://arxiv.org/abs/2102.03150)."""

import torch

from atomgnn.data.data import Batch
from atomgnn.models.utils import (BesselExpansion, compute_edge_vectors_and_norms,
                                  cosine_cutoff, reduce_splits, sum_index)


class PaiNN(torch.nn.Module):
    """Polarizable atom interaction neural network (PaiNN) model.

    PaiNN employs rotationally equivariant atomwise representations.

    Resources:
        PaiNN: https://arxiv.org/abs/2102.03150
    """

    def __init__(
        self,
        node_size: int = 64,
        edge_size: int = 20,
        num_interaction_blocks: int = 3,
        cutoff: float = 5.0,
        pbc: bool = False,
        # Readout
        use_readout: bool = True,
        num_readout_layers: int = 2,
        readout_size: int = 1,
        readout_reduction: str | None = "sum",
        **kwargs,
    ) -> None:
        """Initialize PaiNN model.

        Args:
            node_size: Size of the node state embedding.
            edge_size: Size of the edge state expansion.
            num_interaction_blocks: Number of interaction blocks.
            cutoff: Interaction cutoff distance.
            pbc: Enable periodic boundary conditions (pbc).
            use_readout: Apply readout function.
            num_readout_layers: Number of layers in the readout function.
            readout_size: Output size of the readout function.
            readout_reduction: Reduction applied at the readout, such as 'sum' or 'mean'.
        """
        super().__init__(**kwargs)
        self.node_size = node_size
        self.edge_size = edge_size
        self.num_interaction_blocks = num_interaction_blocks
        self.cutoff = cutoff
        self.pbc = pbc
        self.use_readout = use_readout
        self.num_readout_layers = num_readout_layers
        self.readout_size = readout_size
        self.readout_reduction = readout_reduction
        # Setup node embedding
        num_embeddings = 119  # atomic numbers + 1
        self.node_embedding = torch.nn.Embedding(num_embeddings, node_size)
        # Setup edge expansion
        self.edge_expansion = BesselExpansion(edge_size, cutoff)
        # Setup interaction blocks
        self.interaction_blocks = torch.nn.ModuleList(
            PaiNNInteractionBlock(node_size, edge_size, cutoff)
            for _ in range(num_interaction_blocks))
        # Setup readout network
        if self.use_readout:
            self.readout_net = torch.nn.Sequential()
            for _ in range(num_readout_layers - 1):
                self.readout_net.append(torch.nn.Linear(node_size, node_size))
                self.readout_net.append(torch.nn.SiLU())
            self.readout_net.append(torch.nn.Linear(node_size, readout_size))

    def forward(self, input: Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            input: Batch of input data.
        Returns:
            Tensor of node-level or graph-level predictions.
        """
        # Get scalar node states by embedding node features
        node_states_scalar = self.node_embedding(input.node_features.squeeze())
        # Init vector node states to zero as there is no inititial directional information
        node_states_vector = torch.zeros(
            (node_states_scalar.shape[0], 3, self.node_size),
            dtype=node_states_scalar.dtype,
            device=node_states_scalar.device
        )
        # Compute edge vectors and distances
        edge_vectors, edge_norms = compute_edge_vectors_and_norms(
            input.node_positions, input.edge_index,
            input.edge_shift if self.pbc else None,
            input.cell[input.edge_data_index] if self.pbc else None,
        )
        # Normalise edge vectors (add small number to avoid division by zero)
        edge_vectors = edge_vectors / (edge_norms + 1e-10)
        # Get edge states by expansion of the edge distances
        edge_states = self.edge_expansion(edge_norms)
        # Apply interaction blocks
        for interaction_block in self.interaction_blocks:
            node_states_scalar, node_states_vector = interaction_block(
                node_states_scalar, node_states_vector,
                edge_states, edge_vectors, edge_norms, input.edge_index)
        # Apply node-level readout
        if self.use_readout:
            node_states_scalar = self.readout_net(node_states_scalar)
        # Prepare output
        if self.readout_reduction:
            # Graph-level output
            graph_states_scalar = reduce_splits(
                node_states_scalar, input.num_nodes, reduction=self.readout_reduction
            )
            output_scalar = graph_states_scalar
        else:
            # Node-level output
            output_scalar = node_states_scalar
        # TODO: Compute vector output (e.g. forces as direct output)
        return output_scalar


class PaiNNInteractionBlock(torch.nn.Module):
    """PaiNN interaction block."""

    def __init__(self, node_size: int, edge_size: int, cutoff: float) -> None:
        """Initialize PaiNN interaction block.

        Args:
            node_size: Size of the node states.
            edge_size: Size of the edge states.
            cutoff: Interaction cutoff distance.
        """
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.cutoff = cutoff
        # Message function
        self.edge_filter_net = torch.nn.Linear(edge_size, 3 * node_size)
        self.scalar_message_net = torch.nn.Sequential(
            torch.nn.Linear(node_size, node_size),
            torch.nn.SiLU(),
            torch.nn.Linear(node_size, 3 * node_size),
        )
        # Update function
        self.U_net = torch.nn.Linear(node_size, node_size, bias=False)
        self.V_net = torch.nn.Linear(node_size, node_size, bias=False)
        self.a_net = torch.nn.Sequential(
            torch.nn.Linear(2 * node_size, node_size),
            torch.nn.SiLU(),
            torch.nn.Linear(node_size, 3 * node_size),
        )

    def _message_function(
        self,
        node_states_scalar: torch.Tensor,
        node_states_vector: torch.Tensor,
        edge_states: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_norms: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. 7
        # Roationally-invariant filters (num_edges, 3 * node_size)
        filter_weight = self.edge_filter_net(edge_states)
        # Apply cosine cutoff to filters
        # TODO: The cosine cutoff could be precomputed and input instead of edge_norms
        filter_weight = filter_weight * cosine_cutoff(edge_norms, self.cutoff)
        # Roationally-invariant features (num_nodes, 3 * node_size)
        scalar_output = self.scalar_message_net(node_states_scalar)
        # Select and filter sender nodes (num_edges, 3 * node_size)
        filter_output = filter_weight * scalar_output[edge_index[:, 0]]
        # Split into 3 * (num_edges, node_size)
        gate_nodes, gate_edges, messages_scalar = torch.split(filter_output, self.node_size, dim=1)
        gate_nodes = torch.unsqueeze(gate_nodes, 1)  # (num_edges, 1, node_size)
        gate_edges = torch.unsqueeze(gate_edges, 1)  # (num_edges, 1, node_size)
        # Eq. 8
        # Select and gate sender nodes (num_edges, 3, node_size)
        gated_node_states_vector = node_states_vector[edge_index[:, 0]] * gate_nodes
        gated_edge_vectors = gate_edges * torch.unsqueeze(edge_vectors, 2)
        messages_vector = gated_node_states_vector + gated_edge_vectors
        # Sum messages to compute delta node states
        delta_node_states_scalar_m = sum_index(
            messages_scalar, edge_index[:, 1], torch.zeros_like(node_states_scalar))
        delta_node_states_vector_m = sum_index(
            messages_vector, edge_index[:, 1], torch.zeros_like(node_states_vector))
        # Return delta node states
        return delta_node_states_scalar_m, delta_node_states_vector_m

    def _node_state_update_function(
        self, node_states_scalar: torch.Tensor, node_states_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Linear combinations of features
        Uv = self.U_net(node_states_vector)  # (num_nodes, 3, node_size)
        Vv = self.V_net(node_states_vector)  # (num_nodes, 3, node_size)
        # Note: The original work uses norm here, but we use squared norm to avoid infinite
        # gradients and nans in the training loss due to occasional square root of zero.
        Vv_square_norm = torch.sum(torch.square(Vv), dim=1, keepdim=False)  # (num_nodes, node_size)
        # Scaling functions computed by a shared network (node_size, 3 * node_size)
        a = self.a_net(torch.cat((node_states_scalar, Vv_square_norm), dim=1))
        # Split into 3 * (num_nodes, node_size)
        a_ss, a_sv, a_vv = torch.split(a, self.node_size, dim=1)
        # Eq. 9
        inner_prod_Uv_Vv = torch.sum(Uv * Vv, dim=1)  # (num_nodes, node_size)
        delta_node_states_scalar_u = a_ss + a_sv * inner_prod_Uv_Vv  # (num_nodes, node_size)
        # Eq. 10
        delta_node_states_vector_u = torch.unsqueeze(a_vv, 1) * Uv  # (num_nodes, 3, node_size)
        # Return delta node states
        return delta_node_states_scalar_u, delta_node_states_vector_u

    def forward(
        self,
        node_states_scalar: torch.Tensor,
        node_states_vector: torch.Tensor,
        edge_states: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_norms: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_states_scalar: Tensor with shape (num_nodes, node_size).
            node_states_vector: Tensor with shape (num_nodes, 3, node_size).
            edge_states: Tensor with shape (num_edges, edge_size).
            edge_vectors: Tensor with shape (num_edges, 3).
            edge_norms: Tensor with shape (num_edges, 1).
            edge_index: Tensor with shape (num_edges, 2).
        Returns:
            Tuple of new node_states_scalar and node_states_vector.
        """
        # Compute all messages (the message block in the paper)
        delta_node_states_scalar_m, delta_node_states_vector_m = self._message_function(
            node_states_scalar, node_states_vector,
            edge_states, edge_vectors, edge_norms, edge_index
        )
        # Residual node state update
        node_states_scalar = node_states_scalar + delta_node_states_scalar_m
        node_states_vector = node_states_vector + delta_node_states_vector_m
        # Update node states (the update block in the paper)
        delta_node_states_scalar_u, delta_node_states_vector_u = self._node_state_update_function(
            node_states_scalar, node_states_vector
        )
        # Residual node state update
        node_states_scalar = node_states_scalar + delta_node_states_scalar_u
        node_states_vector = node_states_vector + delta_node_states_vector_u
        # Return updated node states
        return node_states_scalar, node_states_vector
