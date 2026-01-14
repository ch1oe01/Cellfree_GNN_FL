# configs.py
from dataclasses import dataclass

@dataclass
class EnvConfig:
    num_ap: int = 8
    num_ue: int = 64
    num_rb: int = 16
    area_side_m: float = 500.0
    carrier_freq_hz: float = 3.5e9
    num_ap_rx_ant: int = 16
    num_ue_tx_ant: int = 1
    pathloss_exp: float = 3.2
    shadowing_std_db: float = 8.0
    min_dist_m: float = 5.0
    rb_bw_hz: float = 500.0
    noise_psd_w_per_hz: float = 1e-20
    pmax_watt: float = 0.2
    seed: int = 7

@dataclass
class TrainConfig:
    batch_size: int = 8
    steps: int = 2000
    lr: float = 1e-3
    print_every: int = 100

    # 6G-ish knobs
    C: int = 3   # 每個 UE 的協作 AP 數（cluster size）
    L: int = 4   # 每個 RB active UE 上限（同 RB 多 UE）

    # imitation label smoothing (可選)
    label_smoothing: float = 0.0

@dataclass
class ModelConfig:
    hidden: int = 64
    msg_hidden: int = 64
    num_layers: int = 2
