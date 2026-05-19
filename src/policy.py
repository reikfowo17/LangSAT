import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class SmartSATExtractor(BaseFeaturesExtractor):
    """
    Trình trích xuất đặc trưng tùy chỉnh cho hệ thống SmartSAT.
    Xử lý mảng Observation (N_VARS + N_CLAUSES + GRAPH + GLOBAL_FEATURES)
    và nén thành một Feature duy nhất cho PPO.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, n_vars: int = 20, n_clauses: int = 91, n_global: int = 48):
        # Tính toán tổng kích thước đầu vào
        self.n_vars = n_vars
        self.n_clauses = n_clauses
        self.n_global = n_global
        self.graph_size = n_vars * n_clauses
        
        super().__init__(observation_space, features_dim)
        
        # Mạng xử lý riêng cho phần Graph Matrix
        # Chúng ta dùng MLP để xử lý ma trận đã bị flatten
        self.graph_net = nn.Sequential(
            nn.Linear(self.graph_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Mạng xử lý các Đặc trưng Toàn cục (Global Features + Var/Clause Status)
        self.state_net = nn.Sequential(
            nn.Linear(self.n_vars + self.n_clauses + self.n_global, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Tầng tổng hợp (Fusion Layer) gộp 2 luồng thông tin lại
        self.fusion_net = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations có kích thước: (batch_size, OBS_SIZE)
        """
        # 1. Giải nén (Slice) Observation Vector
        idx_var_end = self.n_vars
        idx_clause_end = idx_var_end + self.n_clauses
        idx_graph_end = idx_clause_end + self.graph_size
        
        # Trích xuất từng phần
        var_status = observations[:, :idx_var_end]
        clause_status = observations[:, idx_var_end:idx_clause_end]
        graph_flattened = observations[:, idx_clause_end:idx_graph_end]
        global_feats = observations[:, idx_graph_end:]
        
        # 2. Xử lý Đồ thị
        graph_embedding = self.graph_net(graph_flattened)
        
        # 3. Xử lý Trạng thái động và Đặc trưng toàn cục
        state_input = torch.cat([var_status, clause_status, global_feats], dim=1)
        state_embedding = self.state_net(state_input)
        
        # 4. Gộp thông tin
        fused_input = torch.cat([graph_embedding, state_embedding], dim=1)
        output_features = self.fusion_net(fused_input)
        
        return output_features
