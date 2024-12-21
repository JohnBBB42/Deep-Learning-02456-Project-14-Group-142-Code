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
