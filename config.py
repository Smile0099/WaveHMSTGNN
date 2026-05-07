import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
configs.seed = 5

# -----------------------------------------------------------------------------
# Trainer related
# -----------------------------------------------------------------------------
configs.n_cpu = 1
configs.gpu = 1
if torch.cuda.is_available():
    gpu_id = configs.gpu if torch.cuda.device_count() > configs.gpu else 0
    configs.device = torch.device(f"cuda:{gpu_id}")
else:
    configs.device = torch.device("cpu")
configs.use_gpu = int(torch.cuda.is_available())

configs.batch_size = 8
configs.batch_size_test = 8
configs.lr = 5e-4
configs.weight_decay = 0
configs.display_interval = 42
configs.num_epochs = 1000
configs.early_stopping = True
configs.patience = 20
configs.gradient_clipping = False
configs.clipping_threshold = 1.0

# -----------------------------------------------------------------------------
# Data related
# -----------------------------------------------------------------------------
configs.data_path = "/home/ubuntu/Project(Yu)/data/data_south_ocean.npz"
configs.num_vars = 6
configs.num_nodes = 64
configs.input_dim = configs.num_vars
configs.output_dim = configs.num_vars
configs.nodes = configs.num_nodes * configs.num_vars
configs.enc_in = configs.nodes
configs.dec_in = configs.nodes

# Data variables aligned to the paper:
# SST, SSS, seawater velocity u/v, sea-surface wind stress x/y are used in the paper.
# The order below is the model/channel order used by train_wavehmstgnn.py.
configs.variable_names = [
    "sss",
    "sst",
    "u_current",
    "v_current",
    "wind_stress_x",
    "wind_stress_y",
]
configs.variable_groups = {
    "SSS": [0],
    "SST": [1],
    "Sea water velocity": [2, 3],
    "Sea surface wind stress": [4, 5],
}

configs.seq_len = int(7 * 24 / 6)
configs.pred_len = int(7 * 24 / 6)

# -----------------------------------------------------------------------------
# WaveHMSTGNN model settings
# -----------------------------------------------------------------------------
configs.model_name = "WaveHMSTGNN"
configs.hidden = 32
configs.dropout = 0.1
configs.e_layers = 3
configs.anti_ood = 0
configs.individual = False
configs.embed = "timeF"

# VMWGF: Variable-wise Multi-scale Wavelet Gated Fusion
configs.use_vmwgf = True
configs.scale_number = 4
configs.wavelet_include_detail = True
configs.wavelet_gate_init = 0.0

# HFMU: Harmonic Fusion Memory Unit
configs.use_hfmu = True
configs.hfmu_kernel_size = 3
configs.hfmu_layers = 1
configs.kan_gridsize = 4       # 4/8/16: larger is stronger but slower
configs.kan_input_scale = 1.0  # 1.0 is usually OK for standardized inputs
configs.kan_init_gate = 0.0    # stable start for the KAN branch

# SAGCN: Signed Adaptive Spatial Graph Convolution
configs.use_sagcn = True
configs.nvechidden = 4
configs.sagcn_top_k = 3
configs.sagcn_order = 2

# Backward-compatible names used by the original scripts.
configs.use_ngcn = int(configs.use_sagcn)
configs.use_tgcn = int(configs.use_hfmu)
configs.tvechidden = 4
configs.tk = 10
