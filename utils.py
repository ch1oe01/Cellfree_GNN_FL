import numpy as np
import tensorflow as tf
from dataclasses import dataclass

@dataclass
class SimConfig:
    seed: int = 1234

    # ---------------------------
    # Topology & Geometry
    # ---------------------------
    area_side: float = 300.0
    n_ap: int = 64
    n_ue: int = 40
    bs_height: float = 10.0
    ut_height: float = 1.5

    # Multi-antenna per AP
    M: int = 4

    # ---------------------------
    # Channel Model (Sionna)
    # ---------------------------
    carrier_frequency: float = 3.5e9
    o2i_model: str = "low"
    
    # [必要補充] 為了支援 wirelessNetwork.py 的 OFDM 生成
    subcarrier_spacing: float = 30e3  # 30 kHz
    fft_size: int = 72                # Total subcarriers
    num_time_samples: int = 1         # Snapshot count
    sampling_frequency: float = 30.72e6 

    # [必要補充] 移動性參數 (Mobility)
    ue_speed_mps: float = 1.0
    heading_sigma_rad: float = np.pi / 8
    slot_duration_s: float = 0.5

    # ---------------------------
    # Coherence & Pilots (TDD)
    # ---------------------------
    tau_c: int = 200
    tau_p: int = 16
    p_pilot: float = 0.1  # 導頻功率 (線性值, 建議與 pmax 區分)

    # ---------------------------
    # Resource Allocation (RB / Reuse)
    # ---------------------------
    K: int = 20       # number of RBs
    Q: int = 2        # users per RB capacity (reuse factor)
    TopL: int = 8     # user-centric cluster size (每個 UE 選前幾強的 AP)
    Ca: int = 16      # AP load limit (每個 AP 最多服務幾個 UE)

    # ---------------------------
    # Noise & Power Control
    # ---------------------------
    noise_power_lin: float = 1e-13
    pmax: float = 0.1 # 最大上行發射功率 (線性值, 假設 20dBm = 0.1W)

    # ---------------------------
    # Optimization Algo (WMMSE / AO)
    # ---------------------------
    ao_iters: int = 6
    wmmse_iters: int = 15

    # RB scheduling cost weights (for Heuristic/GNN)
    lambda_pc: float = 3.0   # pilot contamination penalty weight
    lambda_I: float = 0.25   # interference penalty weight

    # Clustering swap
    swap_trials_per_iter: int = 50

    # Monte-Carlo (for stable eval)
    eval_mc: int = 3

def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def cnormal(shape, rng: np.random.Generator, dtype=np.complex64):
    """產生複數高斯雜訊 (Complex Standard Normal)"""
    # 確保標準差為 1 (實部虛部各 1/sqrt(2))
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    x = x / np.sqrt(2.0)
    return x.astype(dtype)

def prelog(cfg: SimConfig) -> float:
    """計算 Pre-log factor (考慮導頻開銷)"""
    return max(0.0, 1.0 - float(cfg.tau_p) / float(cfg.tau_c))

def assert_feasible(cfg: SimConfig):
    """
    檢查參數合理性
    K*Q < U：代表資源不足，系統會只排程 K*Q 個 UE (Blocking Probability > 0)
    """
    if cfg.K <= 0:
        raise ValueError("K (RB數) 必須 > 0")
    if cfg.Q <= 0:
        raise ValueError("Q (Reuse數) 必須 > 0")
    if cfg.n_ue <= 0 or cfg.n_ap <= 0:
        raise ValueError("n_ue / n_ap 必須 > 0")

    if cfg.K * cfg.Q < cfg.n_ue:
        print(f"[Warning] 資源吃緊: 總資源 K*Q={cfg.K*cfg.Q} < 用戶數 U={cfg.n_ue}。")
        print(f"          系統將執行 Admission Control，部分 UE 本次 Slot 無法傳輸。")