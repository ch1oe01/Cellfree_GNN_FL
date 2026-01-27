# runner.py
import time
import numpy as np
from tqdm import tqdm

from utils import SimConfig, set_seed, assert_feasible
from wirelessNetwork import CellFreeNetwork
from baseline_solver import teacher_ao


def run_baseline(cfg: SimConfig, drops: int, show_progress: bool = True):
    """
    跑 baseline teacher AO 多個 drops，回傳：
      mean_sr, std_sr, elapsed_sec, srs(list)
    """
    assert_feasible(cfg)
    set_seed(cfg.seed)

    net = CellFreeNetwork(cfg)

    srs = []
    t0 = time.time()
    it = range(drops)
    if show_progress:
        it = tqdm(it, desc=f"Simulating drops (seed={cfg.seed})")

    for _ in it:
        _ue_pos, _ap_pos, beta, pilot_id, H, Hhat = net.generate_drop()
        _A, _S, _p, _C, _n, sr = teacher_ao(beta, pilot_id, H, Hhat, cfg)
        srs.append(float(sr))

    elapsed = time.time() - t0
    srs = np.array(srs, dtype=np.float32)
    return float(srs.mean()), float(srs.std()), float(elapsed), srs.tolist()
