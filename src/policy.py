import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmartSATGraphExtractor(BaseFeaturesExtractor):
    """Bipartite clause-variable message passing for SmartSAT PPO."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        n_vars: int = 20,
        n_clauses: int = 91,
        n_global: int = 48,
        hidden_dim: int = 96,
        message_rounds: int = 2,
    ):
        super().__init__(observation_space, features_dim)
        self.n_vars = n_vars
        self.n_clauses = n_clauses
        self.n_global = n_global
        self.graph_size = n_vars * n_clauses
        self.hidden_dim = hidden_dim
        self.message_rounds = message_rounds

        self.var_ids = nn.Embedding(n_vars, hidden_dim)
        self.clause_ids = nn.Embedding(n_clauses, hidden_dim)
        self.global_net = nn.Sequential(
            nn.Linear(n_global, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.var_init = nn.Sequential(
            nn.Linear(1 + hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.clause_init = nn.Sequential(
            nn.Linear(1 + hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pos_var_to_clause = nn.Linear(hidden_dim, hidden_dim)
        self.neg_var_to_clause = nn.Linear(hidden_dim, hidden_dim)
        self.pos_clause_to_var = nn.Linear(hidden_dim, hidden_dim)
        self.neg_clause_to_var = nn.Linear(hidden_dim, hidden_dim)
        self.clause_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
        )
        self.var_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        var_status, clause_status, graph, global_features = self._split_observation(observations)
        batch_size = observations.shape[0]
        device = observations.device

        global_embedding = self.global_net(global_features)
        var_ids = self.var_ids(torch.arange(self.n_vars, device=device)).unsqueeze(0).expand(batch_size, -1, -1)
        clause_ids = (
            self.clause_ids(torch.arange(self.n_clauses, device=device))
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        var_global = global_embedding.unsqueeze(1).expand(-1, self.n_vars, -1)
        clause_global = global_embedding.unsqueeze(1).expand(-1, self.n_clauses, -1)

        var_h = self.var_init(torch.cat([var_status.unsqueeze(-1), var_ids, var_global], dim=-1))
        clause_h = self.clause_init(torch.cat([clause_status.unsqueeze(-1), clause_ids, clause_global], dim=-1))

        pos_edges = (graph > 0).to(observations.dtype)
        neg_edges = (graph < 0).to(observations.dtype)
        for _ in range(self.message_rounds):
            clause_h = self._update_clauses(clause_h, var_h, pos_edges, neg_edges)
            var_h = self._update_vars(var_h, clause_h, pos_edges, neg_edges)

        pooled_vars = var_h.mean(dim=1)
        pooled_clauses = clause_h.mean(dim=1)
        return self.fusion(torch.cat([pooled_vars, pooled_clauses, global_embedding], dim=-1))

    def _split_observation(self, observations: torch.Tensor):
        var_end = self.n_vars
        clause_end = var_end + self.n_clauses
        graph_end = clause_end + self.graph_size
        var_status = observations[:, :var_end]
        clause_status = observations[:, var_end:clause_end]
        graph = observations[:, clause_end:graph_end].reshape(-1, self.n_clauses, self.n_vars)
        global_features = observations[:, graph_end:graph_end + self.n_global]
        return var_status, clause_status, graph, global_features

    def _update_clauses(
        self,
        clause_h: torch.Tensor,
        var_h: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> torch.Tensor:
        pos_degree = pos_edges.sum(dim=2, keepdim=True).clamp_min(1.0)
        neg_degree = neg_edges.sum(dim=2, keepdim=True).clamp_min(1.0)
        pos_msg = torch.bmm(pos_edges, var_h) / pos_degree
        neg_msg = torch.bmm(neg_edges, var_h) / neg_degree
        return self.clause_update(
            torch.cat(
                [
                    clause_h,
                    self.pos_var_to_clause(pos_msg),
                    self.neg_var_to_clause(neg_msg),
                ],
                dim=-1,
            )
        )

    def _update_vars(
        self,
        var_h: torch.Tensor,
        clause_h: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> torch.Tensor:
        pos_edges_t = pos_edges.transpose(1, 2)
        neg_edges_t = neg_edges.transpose(1, 2)
        pos_degree = pos_edges_t.sum(dim=2, keepdim=True).clamp_min(1.0)
        neg_degree = neg_edges_t.sum(dim=2, keepdim=True).clamp_min(1.0)
        pos_msg = torch.bmm(pos_edges_t, clause_h) / pos_degree
        neg_msg = torch.bmm(neg_edges_t, clause_h) / neg_degree
        return self.var_update(
            torch.cat(
                [
                    var_h,
                    self.pos_clause_to_var(pos_msg),
                    self.neg_clause_to_var(neg_msg),
                ],
                dim=-1,
            )
        )


def policy_kwargs(features_dim: int = 128) -> dict:
    return {
        "features_extractor_class": SmartSATGraphExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "n_global": 48,
        },
    }
