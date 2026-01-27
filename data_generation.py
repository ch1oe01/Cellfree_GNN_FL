# data_generation.py
import os
import numpy as np
from tqdm import tqdm
from utils import SimConfig, set_seed, assert_feasible
from wirelessNetwork import CellFreeNetwork
from baseline_solver import teacher_ao

def generate_dataset(out_path: str, drops: int = 200, cfg: SimConfig = SimConfig()):
    assert_feasible(cfg)
    set_seed(cfg.seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    net = CellFreeNetwork(cfg)

    betas, pilots, As, Ss, ps, srs = [], [], [], [], [], []

    for _ in tqdm(range(drops), desc="Generating teacher dataset"):
        _ue_pos, _ap_pos, beta, pilot_id, H, Hhat = net.generate_drop()
        A_mat, S, p, _C, _n_eff, sr = teacher_ao(beta, pilot_id, H, Hhat, cfg)

        betas.append(beta.astype(np.float32))
        pilots.append(pilot_id.astype(np.int32))
        As.append(A_mat.astype(np.int32))
        Ss.append(S.astype(np.int32))
        ps.append(p.astype(np.float32))
        srs.append(float(sr))

    np.savez(out_path,
             beta=np.stack(betas),        # [N,A,U]
             pilot=np.stack(pilots),      # [N,U]
             A=np.stack(As),              # [N,U,A]
             S=np.stack(Ss),              # [N,U,K]
             p=np.stack(ps),              # [N,U]
             sum_rate=np.array(srs, dtype=np.float32))
    print(f"Saved -> {out_path}")
    print(f"Avg SR: {np.mean(srs):.3f} ± {np.std(srs):.3f}")

if __name__ == "__main__":
    cfg = SimConfig()
    generate_dataset("./cf_data/teacher_v1_dataset.npz", drops=200, cfg=cfg)
