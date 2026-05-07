"""
WaveHMSTGNN implementation aligned with the paper structure.

Paper module names used in this file:
- VMWGF: Variable-wise Multi-scale Wavelet Gated Fusion
- HFMU: Harmonic Fusion Memory Unit
- SAGCN: Signed Adaptive Spatial Graph Convolution
- WaveHMSTGNN: Wavelet-enhanced Harmonic Memory-based Spatio-Temporal Adaptive Graph Neural Network
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from config import configs
except Exception:  # pragma: no cover - allows importing the model without config.py
    configs = None

_SQRT1_2 = 0.7071067811865476  # 1 / sqrt(2)


# -----------------------------------------------------------------------------
# Haar DWT utilities used by VMWGF
# -----------------------------------------------------------------------------
def haar_dwt_multilevel(x: torch.Tensor, max_level: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Multi-level Haar DWT along the time dimension.

    Args:
        x: Tensor with shape [B, T, C] or [B, T, N].
        max_level: Number of Haar decomposition levels.

    Returns:
        approx_list: [a_1, a_2, ..., a_L], where a_l has time length ceil(T / 2^l).
        detail_list: [d_1, d_2, ..., d_L], same shapes as approx_list.
    """
    approx = x
    approx_list, detail_list = [], []

    for _ in range(max_level):
        time_length = approx.shape[1]
        if time_length % 2 == 1:
            # Odd length: duplicate the final time step for Haar pairing.
            approx = torch.cat([approx, approx[:, -1:, :]], dim=1)

        even = approx[:, 0::2, :]
        odd = approx[:, 1::2, :]
        next_approx = (even + odd) * _SQRT1_2
        next_detail = (even - odd) * _SQRT1_2

        approx_list.append(next_approx)
        detail_list.append(next_detail)
        approx = next_approx

    return approx_list, detail_list


def upsample_time_coefficients(y: torch.Tensor, target_time_length: int) -> torch.Tensor:
    """
    Align multi-scale coefficients back to the original time resolution.

    Args:
        y: Tensor with shape [B, L, C] or [B, L, N].
        target_time_length: Target time length T.

    Returns:
        Tensor with shape [B, T, C] or [B, T, N].
    """
    y = y.permute(0, 2, 1)  # [B, C/N, L]
    y = F.interpolate(y, size=target_time_length, mode="linear", align_corners=False)
    return y.permute(0, 2, 1).contiguous()


class VariableWiseMultiScaleWaveletGatedFusion(nn.Module):
    """
    VMWGF: Variable-wise Multi-scale Wavelet Gated Fusion.

    For each variable, this module applies multi-level Haar DWT, aligns low-frequency
    approximation and high-frequency detail coefficients back to the original time
    resolution, and fuses them through tanh-bounded residual gates.

    Input:  [B, T, N, V]
    Output: [B, T, N, V]
    """

    def __init__(
        self,
        num_vars: int,
        max_level: int,
        include_detail: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.num_vars = int(num_vars)
        self.max_level = max(1, int(max_level))
        self.include_detail = bool(include_detail)

        # Gate depends on variable v and level l only: O(V * L).
        # gate_init=0 makes the initial module close to identity mapping.
        self.approx_gate = nn.Parameter(torch.zeros(self.num_vars, self.max_level) + float(gate_init))
        if self.include_detail:
            self.detail_gate = nn.Parameter(torch.zeros(self.num_vars, self.max_level) + float(gate_init))
        else:
            self.register_parameter("detail_gate", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_length, num_nodes, num_vars = x.shape
        assert num_vars == self.num_vars, f"Expected V={self.num_vars}, got V={num_vars}"

        fused_variables = []
        for var_idx in range(num_vars):
            x_var = x[..., var_idx]  # [B, T, N]
            approx_list, detail_list = haar_dwt_multilevel(x_var, self.max_level)

            fused = x_var
            for level in range(1, self.max_level + 1):
                approx_up = upsample_time_coefficients(approx_list[level - 1], time_length)
                approx_weight = torch.tanh(self.approx_gate[var_idx, level - 1])
                fused = fused + approx_weight * approx_up

                if self.include_detail:
                    detail_up = upsample_time_coefficients(detail_list[level - 1], time_length)
                    detail_weight = torch.tanh(self.detail_gate[var_idx, level - 1])
                    fused = fused + detail_weight * detail_up

            fused_variables.append(fused.unsqueeze(-1))

        return torch.cat(fused_variables, dim=-1)


# Short alias matching the paper abbreviation.
VMWGF = VariableWiseMultiScaleWaveletGatedFusion


# -----------------------------------------------------------------------------
# Fourier-parameterized KAN branch used by HFMU
# -----------------------------------------------------------------------------
class FourierKANLinear(nn.Module):
    """
    Fourier-parameterized KAN mapping used in the HFMU KAN convolution branch.

    The module follows the paper description: layer normalization, learnable
    per-dimension input scale, learnable frequency/phase bases, frequency decay,
    and an optional linear shortcut.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        gridsize: int = 8,
        addbias: bool = True,
        input_scale: float = 1.0,
        chunk_size: int = 2048,
        use_layernorm: bool = True,
        use_base_linear: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.gridsize = int(gridsize)
        self.addbias = bool(addbias)
        self.chunk_size = int(chunk_size)

        self.in_norm = nn.LayerNorm(self.in_dim) if use_layernorm else None
        self.base = nn.Linear(self.in_dim, self.out_dim, bias=False) if use_base_linear else None

        if isinstance(input_scale, (float, int)):
            self.input_scale = nn.Parameter(torch.ones(self.in_dim) * float(input_scale))
        else:
            self.input_scale = nn.Parameter(torch.tensor(input_scale, dtype=torch.float32))

        self.fourier_coefficients = nn.Parameter(
            torch.randn(2, self.out_dim, self.in_dim, self.gridsize)
            / (math.sqrt(self.in_dim) * math.sqrt(self.gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(self.out_dim))
        else:
            self.register_parameter("bias", None)

        omega0 = torch.arange(1, self.gridsize + 1, dtype=torch.float32).view(1, 1, self.gridsize)
        omega0 = omega0.repeat(1, self.in_dim, 1)
        self.log_omega = nn.Parameter(torch.log(omega0))
        self.phase = nn.Parameter(torch.zeros(1, self.in_dim, self.gridsize))

        decay = 1.0
        self.register_buffer(
            "frequency_decay",
            (1.0 / (torch.arange(1, self.gridsize + 1, dtype=torch.float32) ** decay)).view(1, 1, -1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_dim)

        if self.in_norm is not None:
            x = self.in_norm(x)
        x = x * self.input_scale

        coeff_cos = self.fourier_coefficients[0]
        coeff_sin = self.fourier_coefficients[1]
        omega = torch.exp(self.log_omega).to(x.device)
        frequency_decay = self.frequency_decay.to(x.device)

        outputs = []
        for start in range(0, x.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, x.shape[0])
            x_chunk = x[start:end]
            x_frequency = x_chunk.unsqueeze(-1) * omega + self.phase

            cos_basis = torch.cos(x_frequency)
            sin_basis = torch.sin(x_frequency)

            y_cos = torch.einsum("mig,oig->mo", cos_basis, coeff_cos * frequency_decay)
            y_sin = torch.einsum("mig,oig->mo", sin_basis, coeff_sin * frequency_decay)
            y = y_cos + y_sin

            if self.base is not None:
                y = y + self.base(x_chunk)
            if self.bias is not None:
                y = y + self.bias
            outputs.append(y)

        y = torch.cat(outputs, dim=0)
        return y.reshape(*original_shape[:-1], self.out_dim)


class KANConv1d(nn.Module):
    """
    KAN version of Conv1d implemented by unfolding node-dimension patches.

    Input:  [B, C_in, N]
    Output: [B, C_out, N]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        gridsize: int = 8,
        addbias: bool = True,
        input_scale: float = 1.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd to keep the node length unchanged."
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2 if padding is None else int(padding)

        patch_dim = self.in_channels * self.kernel_size
        self.kan = FourierKANLinear(
            in_dim=patch_dim,
            out_dim=self.out_channels,
            gridsize=gridsize,
            addbias=addbias,
            input_scale=input_scale,
            chunk_size=1024,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, num_nodes = x.shape
        x = F.pad(x, (self.padding, self.padding))
        patches = x.unfold(dimension=2, size=self.kernel_size, step=1)  # [B, C_in, N, K]
        patches = patches.permute(0, 2, 1, 3).contiguous().reshape(
            batch_size * num_nodes, in_channels * self.kernel_size
        )
        y = self.kan(patches).reshape(batch_size, num_nodes, self.out_channels)
        return y.permute(0, 2, 1).contiguous()


class HFMUCell(nn.Module):
    """
    HFMU cell: Harmonic Fusion Memory Unit cell.

    It fuses a conventional Conv1d gate branch and a Fourier-KAN Conv1d gate
    branch with channel-level sigmoid-bounded fusion coefficients.

    x: [B, C_in, N]
    h: [B, C_h,  N]
    c: [B, C_h,  N]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
        kan_gridsize: int = 8,
        kan_input_scale: float = 1.0,
        kan_init_gate: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd to keep the node length unchanged."
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        padding = kernel_size // 2

        self.conv_gate = nn.Conv1d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.kan_gate = KANConv1d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            gridsize=kan_gridsize,
            addbias=bias,
            input_scale=kan_input_scale,
        )

        self.harmonic_fusion_gate = nn.Parameter(torch.ones(4 * self.hidden_dim, 1) * float(kan_init_gate))

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)  # [B, C_in + C_h, N]

        gates_conv = self.conv_gate(combined)
        if self.training:
            gates_kan = checkpoint(self.kan_gate, combined, use_reentrant=False)
        else:
            gates_kan = self.kan_gate(combined)

        alpha = torch.sigmoid(self.harmonic_fusion_gate)  # [4 * C_h, 1]
        gates = gates_conv + alpha * gates_kan

        input_gate, forget_gate, output_gate, candidate_state = torch.chunk(gates, 4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate_state = torch.tanh(candidate_state)

        c_next = forget_gate * c + input_gate * candidate_state
        h_next = output_gate * torch.tanh(c_next)
        return h_next, c_next


class HarmonicFusionMemoryUnit(nn.Module):
    """
    HFMU: Harmonic Fusion Memory Unit.

    Recursive temporal update over T. At every step, HFMU applies local node
    convolution through a conventional Conv1d branch and a Fourier-KAN branch.

    Input:  [B, T, N, C]
    Output: [B, T, N, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
        kan_gridsize: int = 8,
        kan_input_scale: float = 1.0,
        kan_init_gate: float = 0.0,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        cells = []
        for layer_idx in range(self.num_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(
                HFMUCell(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    bias=True,
                    kan_gridsize=kan_gridsize,
                    kan_input_scale=kan_input_scale,
                    kan_init_gate=kan_init_gate,
                )
            )
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_length, num_nodes, _ = x.shape
        layer_input = x

        for layer_idx, cell in enumerate(self.cells):
            h = torch.zeros(batch_size, self.hidden_dim, num_nodes, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_dim, num_nodes, device=x.device, dtype=x.dtype)

            outputs = []
            for time_idx in range(time_length):
                x_t = layer_input[:, time_idx].permute(0, 2, 1)  # [B, C_in, N]
                h, c = cell(x_t, h, c)
                outputs.append(h)

            layer_output = torch.stack(outputs, dim=1)  # [B, T, C_h, N]
            layer_output = layer_output.permute(0, 1, 3, 2).contiguous()  # [B, T, N, C_h]

            if self.dropout > 0 and layer_idx < self.num_layers - 1:
                layer_output = F.dropout(layer_output, p=self.dropout, training=self.training)
            layer_input = layer_output

        return layer_input


# Short alias matching the paper abbreviation.
HFMU = HarmonicFusionMemoryUnit


# -----------------------------------------------------------------------------
# Signed Adaptive Spatial Graph Convolution used by WaveHMSTGNN
# -----------------------------------------------------------------------------
class SignedAdaptiveSpatialGraphConvolution(nn.Module):
    """
    SAGCN: Signed Adaptive Spatial Graph Convolution.

    This module learns node embeddings, builds a signed sparse adaptive adjacency
    using Top-k/Bottom-k masks, and applies multi-order node diffusion.

    Input:  [B, T, N, C]
    Output: [B, T, N, C]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        embedding_dim: int,
        dropout: float,
        diffusion_order: int = 2,
        top_k: int = 3,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.dropout = float(dropout)
        self.diffusion_order = int(diffusion_order)
        self.top_k = int(top_k)

        self.node_embedding_left = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim, device=device))
        self.node_embedding_right = nn.Parameter(torch.randn(self.embedding_dim, self.num_nodes, device=device))

        self.projection = nn.Linear((self.diffusion_order + 1) * self.hidden_dim, self.hidden_dim)
        self.activation = nn.GELU()

    @staticmethod
    def _masked_softmax(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        masked_values = values.masked_fill(~mask, torch.finfo(values.dtype).min)
        weights = F.softmax(masked_values, dim=dim)
        return weights * mask.to(values.dtype)

    def get_signed_sparse_adjacency(self) -> torch.Tensor:
        dense_adj = F.relu(self.node_embedding_left @ self.node_embedding_right)  # [N, N]
        num_nodes = dense_adj.shape[-1]
        k = max(1, min(self.top_k, num_nodes))

        top_indices = torch.topk(dense_adj, k=k, dim=-1, largest=True).indices
        bottom_indices = torch.topk(dense_adj, k=k, dim=-1, largest=False).indices

        positive_mask = torch.zeros_like(dense_adj, dtype=torch.bool)
        negative_mask = torch.zeros_like(dense_adj, dtype=torch.bool)
        positive_mask.scatter_(dim=-1, index=top_indices, value=True)
        negative_mask.scatter_(dim=-1, index=bottom_indices, value=True)

        positive_weights = self._masked_softmax(dense_adj, positive_mask, dim=-1)
        negative_source = 1.0 / (dense_adj + 1.0)
        negative_weights = -self._masked_softmax(negative_source, negative_mask, dim=-1)

        return positive_weights + negative_weights

    @staticmethod
    def diffuse(x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C], adjacency: [N, N]
        return torch.einsum("btnc,nm->btmc", x, adjacency).contiguous()

    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        if adjacency is None:
            adjacency = self.get_signed_sparse_adjacency()

        diffusion_outputs = [x]
        x_order = x
        for _ in range(self.diffusion_order):
            x_order = self.diffuse(x_order, adjacency)
            diffusion_outputs.append(x_order)

        h = torch.cat(diffusion_outputs, dim=-1)
        h = self.projection(h)
        h = self.activation(h)
        return F.dropout(h, self.dropout, training=self.training)


# Short alias matching the paper abbreviation.
SAGCN = SignedAdaptiveSpatialGraphConvolution


# -----------------------------------------------------------------------------
# WaveHMSTGNN block and full model
# -----------------------------------------------------------------------------
class WaveHMSTGNNBlock(nn.Module):
    """
    A single WaveHMSTGNN encoder block:
    VMWGF -> variable projection -> HFMU -> SAGCN -> readout.

    Input:  [B, T, N, V]
    Output: [B, T, N, V]
    """

    def __init__(self, cfg):
        super().__init__()
        self.use_vmwgf = bool(getattr(cfg, "use_vmwgf", True))
        self.use_hfmu = bool(getattr(cfg, "use_hfmu", getattr(cfg, "use_tgcn", True)))
        self.use_sagcn = bool(getattr(cfg, "use_sagcn", getattr(cfg, "use_ngcn", True)))
        self.seq_len = int(cfg.seq_len)
        self.dropout = float(cfg.dropout)

        self.num_vars = int(getattr(cfg, "num_vars", 6))
        enc_in = int(getattr(cfg, "enc_in", 384))
        self.num_nodes = int(getattr(cfg, "num_nodes", enc_in // self.num_vars))
        assert self.num_nodes * self.num_vars == enc_in, f"enc_in={enc_in} != num_nodes*num_vars"

        self.hidden_dim = int(cfg.hidden)

        max_level = max(1, int(math.log2(self.seq_len)))
        max_level = min(max_level, int(getattr(cfg, "scale_number", max_level)))

        self.vmwgf = VMWGF(
            num_vars=self.num_vars,
            max_level=max_level,
            include_detail=bool(getattr(cfg, "wavelet_include_detail", True)),
            gate_init=float(getattr(cfg, "wavelet_gate_init", 0.0)),
        )

        self.variable_projection = nn.Linear(self.num_vars, self.hidden_dim)

        self.hfmu = HFMU(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=int(getattr(cfg, "hfmu_kernel_size", 3)),
            num_layers=int(getattr(cfg, "hfmu_layers", 1)),
            dropout=self.dropout,
            kan_gridsize=int(getattr(cfg, "kan_gridsize", 8)),
            kan_input_scale=float(getattr(cfg, "kan_input_scale", 1.0)),
            kan_init_gate=float(getattr(cfg, "kan_init_gate", 0.0)),
        )

        self.sagcn = SAGCN(
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            embedding_dim=int(getattr(cfg, "nvechidden", getattr(cfg, "sagcn_embedding_dim", 4))),
            dropout=self.dropout,
            diffusion_order=int(getattr(cfg, "sagcn_order", 2)),
            top_k=int(getattr(cfg, "sagcn_top_k", 3)),
            device=getattr(cfg, "device", None),
        )

        self.readout = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.num_vars),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_vmwgf:
            x = self.vmwgf(x)

        projected = self.variable_projection(x)  # [B, T, N, hidden_dim]
        hidden = projected

        if self.use_hfmu:
            hidden = self.hfmu(hidden) + hidden

        if self.use_sagcn:
            hidden = self.sagcn(hidden) + hidden

        output = torch.cat([projected, hidden], dim=-1)
        output = self.readout(output)
        return F.dropout(output, p=self.dropout, training=self.training)


class WaveHMSTGNN(nn.Module):
    """
    Wavelet-enhanced Harmonic Memory-based Spatio-Temporal Adaptive GNN.

    Input:  [B, seq_len, enc_in], where enc_in = num_nodes * num_vars
    Output: [B, pred_len, enc_in]
    """

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.seq_len)
        self.pred_len = int(cfg.pred_len)
        self.num_layers = int(cfg.e_layers)
        self.anti_ood = bool(getattr(cfg, "anti_ood", False))

        self.num_vars = int(getattr(cfg, "num_vars", 6))
        self.enc_in = int(getattr(cfg, "enc_in", 384))
        self.num_nodes = int(getattr(cfg, "num_nodes", self.enc_in // self.num_vars))
        assert self.num_nodes * self.num_vars == self.enc_in, (
            f"enc_in={self.enc_in} != num_nodes*num_vars={self.num_nodes * self.num_vars}"
        )

        self.encoder_blocks = nn.ModuleList([WaveHMSTGNNBlock(cfg) for _ in range(self.num_layers)])

        # One time projection head per variable, matching the variable-wise design.
        self.time_projection = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.num_vars)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_length, feature_dim = x.shape
        assert time_length == self.seq_len, f"Expected seq_len={self.seq_len}, got {time_length}"
        assert feature_dim == self.enc_in, f"Expected enc_in={self.enc_in}, got {feature_dim}"

        x = x.view(batch_size, time_length, self.num_nodes, self.num_vars)

        if self.anti_ood:
            seq_last = x[:, -1:, :, :].detach()
            x = x - seq_last
        else:
            seq_last = None

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        predictions = []
        for var_idx in range(self.num_vars):
            x_var = x[..., var_idx]  # [B, T, N]
            pred_var = self.time_projection[var_idx](x_var.permute(0, 2, 1)).permute(0, 2, 1)
            predictions.append(pred_var.unsqueeze(-1))

        prediction = torch.cat(predictions, dim=-1)  # [B, pred_len, N, V]

        if seq_last is not None:
            prediction = prediction + seq_last

        return prediction.reshape(batch_size, self.pred_len, feature_dim)




if __name__ == "__main__":
    if configs is None:
        raise RuntimeError("config.py was not found. Please run this file with a valid config.py.")

    sample_x = torch.randn(32, configs.seq_len, configs.enc_in, device=configs.device)
    model = WaveHMSTGNN(configs).to(configs.device)
    sample_y = model(sample_x)
    print("input shape:", sample_x.shape)
    print("output shape:", sample_y.shape)
