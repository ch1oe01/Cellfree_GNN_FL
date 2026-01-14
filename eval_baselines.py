# eval_baselines.py
import tensorflow as tf
from configs import EnvConfig, TrainConfig
from channel_env import SionnaCellFreeUplinkEnv

def main():
    env_cfg = EnvConfig()
    tr_cfg = TrainConfig()
    env = SionnaCellFreeUplinkEnv(**env_cfg.__dict__)

    B = 1
    trials = 50

    sr_rand = 0.0
    sr_top = 0.0

    for t in range(trials):
        h, g, d = env.sample_channel(batch_size=B)
        ap_idx_uc = env.get_clusters_nearest(d, C=tr_cfg.C)

        x_r = env.schedule_random_L(B=B, U=env.U, K=env.K, L=tr_cfg.L, seed=100 + t)
        x_t = env.schedule_topL_by_cluster_gain(h, ap_idx_uc, L=tr_cfg.L)

        srr, _, _ = env.compute_sum_rate_mrc_clustered(h, x_r, ap_idx_uc)
        srt, _, _ = env.compute_sum_rate_mrc_clustered(h, x_t, ap_idx_uc)

        sr_rand += float(srr.numpy())
        sr_top += float(srt.numpy())

    print("=== Baseline eval ===")
    print(f"Trials: {trials}, A={env.A}, U={env.U}, K={env.K}, C={tr_cfg.C}, L={tr_cfg.L}")
    print(f"Avg SR Random: {sr_rand/trials:.3f} bps")
    print(f"Avg SR Top-L : {sr_top/trials:.3f} bps")

if __name__ == "__main__":
    main()
