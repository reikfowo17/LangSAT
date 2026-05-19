import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class TrueGNNSATExtractor(BaseFeaturesExtractor):
    """
    Kiến trúc Bipartite GNN chuẩn mực cho bài toán SAT (Lấy cảm hứng từ NeuroSAT).
    Thực hiện Message Passing giữa Đỉnh Biến (Variables) và Đỉnh Mệnh đề (Clauses).
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, 
                 n_vars: int = 20, n_clauses: int = 91, n_global: int = 48, gnn_iters: int = 3):
        
        self.n_vars = n_vars
        self.n_clauses = n_clauses
        self.n_global = n_global
        self.graph_size = n_vars * n_clauses
        self.gnn_iters = gnn_iters # Số vòng lặp trao đổi thông tin
        self.hidden_dim = 64
        
        super().__init__(observation_space, features_dim)
        
        # 1. Các lớp nhúng (Embedding) ban đầu
        self.var_emb = nn.Linear(1, self.hidden_dim)     # Trạng thái của biến
        self.clause_emb = nn.Linear(1, self.hidden_dim)  # Trạng thái của mệnh đề
        
        # 2. Mạng cập nhật Đỉnh (Node Update Networks)
        # Cập nhật Mệnh đề dựa trên thông điệp từ Biến chuyển tới
        self.clause_update = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Cập nhật Biến dựa trên thông điệp từ Mệnh đề + Biến đối ngẫu (Flip)
        self.var_update = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), # *2 vì nhận cả biến phủ định
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 3. Lớp nén cuối cùng kết hợp Đặc trưng toàn cục
        self.final_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.n_global, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # --- BƯỚC 1: GIẢI NÉN ĐỒ THỊ ---
        idx_var = self.n_vars
        idx_clause = idx_var + self.n_clauses
        idx_graph = idx_clause + self.graph_size
        
        var_status = observations[:, :idx_var].unsqueeze(-1)          # (Batch, N, 1)
        clause_status = observations[:, idx_var:idx_clause].unsqueeze(-1) # (Batch, M, 1)
        global_feats = observations[:, idx_graph:]                    # (Batch, Global)
        
        # Phục hồi Ma trận kề (Adjacency Matrix): Kích thước (Batch, M, N)
        graph_matrix = observations[:, idx_clause:idx_graph].view(batch_size, self.n_clauses, self.n_vars)
        
        # Tách đồ thị thành 2 loại cạnh: Biến dương (1) và Biến âm (-1)
        # Dùng ReLU để lọc: relu(x) giữ 1 bỏ -1. relu(-x) biến -1 thành 1, bỏ 1.
        E_pos = torch.relu(graph_matrix)  
        E_neg = torch.relu(-graph_matrix) 
        
        # --- BƯỚC 2: KHỞI TẠO ĐỈNH (INIT NODES) ---
        # Khởi tạo vector nhúng
