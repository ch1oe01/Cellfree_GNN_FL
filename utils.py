# utils.py
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

@dataclass
class SimConfig:
    seed: int = 1234

    # Topology
    area_side: float = 300.0
    n_ap: int = 32
    n_ue: int = 40
    bs_height: float = 10.0
    ut_height: float = 1.5

    # Multi-antenna per AP
    M: int = 4

    # Channel (Sionna large-scale)
    carrier_frequency: float = 3.5e9
    o2i_model: str = "low"

    # Coherence/pilots (TDD)
    tau_c: int = 200
    tau_p: int = 10
    p_pilot: float = 1.0

    # RB / reuse
    K: int = 20     # number of RBs
    Q: int = 2      # users per RB capacity (reuse)
    TopL: int = 8   # user-centric cluster size
    Ca: int = 16    # AP load limit (#served UEs)

    # Noise / power
    noise_power_lin: float = 1e-13
    pmax: float = 1.0

    # AO & WMMSE
    ao_iters: int = 6
    wmmse_iters: int = 15

    # RB scheduling cost weights
    lambda_pc: float = 3.0   # pilot contamination penalty weight
    lambda_I: float = 0.25   # interference penalty weight

    # Clustering swap
    swap_trials_per_iter: int = 50

    # Monte-Carlo (for more stable eval; keep small to run fast)
    eval_mc: int = 3

def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def cnormal(shape, rng: np.random.Generator, dtype=np.complex64):
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    x = x / np.sqrt(2.0)
    return x.astype(dtype)

def prelog(cfg: SimConfig) -> float:
    return max(0.0, 1.0 - float(cfg.tau_p) / float(cfg.tau_c))

def assert_feasible(cfg: SimConfig):
    """
    允許 K*Q < U：代表資源不足，系統會只排程 K*Q 個 UE，其餘 UE 本 slot 不傳 (S=0)。
    這比「強迫塞爆 RB」更符合現實的 MAC scheduling。
    """
    if cfg.K <= 0:
        raise ValueError("K 必須 > 0")
    if cfg.Q <= 0:
        raise ValueError("Q 必須 > 0")
    if cfg.n_ue <= 0 or cfg.n_ap <= 0:
        raise ValueError("n_ue / n_ap 必須 > 0")

    if cfg.K * cfg.Q < cfg.n_ue:
        print(f"[Warning] K*Q={cfg.K*cfg.Q} < U={cfg.n_ue}：將只排程 K*Q 個 UE，其餘本 slot 不傳。")

