# baseline_solver.py
import numpy as np
import math
from utils import SimConfig, prelog

# -----------------------
# Clustering (Top-L + AP load) + Swap improvement (heuristic)
# -----------------------
def init_topL_clustering(beta: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """A_mat[u,a] in {0,1}, shape [U,A]"""
    A, U = cfg.n_ap, cfg.n_ue
    A_mat = np.zeros((U, A), dtype=np.int32)

    for u in range(U):
        idx = np.argsort(-beta[:, u])[: cfg.TopL]
        A_mat[u, idx] = 1

    # AP load repair
    load = A_mat.sum(axis=0)
    for a in range(A):
        while load[a] > cfg.Ca:
            us = np.where(A_mat[:, a] == 1)[0]
            u_drop = us[np.argmin(beta[a, us])]
            A_mat[u_drop, a] = 0
            load[a] -= 1

    # ensure each UE has >=1 AP
    for u in range(U):
        if A_mat[u].sum() == 0:
            a_best = int(np.argmax(beta[:, u]))
            A_mat[u, a_best] = 1

    return A_mat

def clustering_swap_improve(beta: np.ndarray, A_mat: np.ndarray, cfg: SimConfig, rng: np.random.Generator):
    """
    研究用：在不重算所有 combining 的情況下，做一個「可跑且有效」的 swap heuristic
    目標：提高每 UE 的 signal proxy 並降低對其他 UE 的總干擾 proxy
      utility(u,a) = beta[a,u] - lambda_I * sum_{v != u} beta[a,v]
    """
    A, U = cfg.n_ap, cfg.n_ue
    load = A_mat.sum(axis=0).astype(np.int32)

    # precompute total beta per AP
    beta_sum_ap = np.sum(beta, axis=1)  # [A]

    for _ in range(cfg.swap_trials_per_iter):
        u = int(rng.integers(0, U))
        served_aps = np.where(A_mat[u] == 1)[0]
        if len(served_aps) == 0:
            continue
        a_old = int(served_aps[rng.integers(0, len(served_aps))])

        # candidate APs: top candidates not in cluster
        cand = np.argsort(-beta[:, u])[: min(A, cfg.TopL * 3)]
        cand = [int(a) for a in cand if A_mat[u, a] == 0]
        if len(cand) == 0:
            continue
        a_new = int(cand[rng.integers(0, len(cand))])

        if load[a_new] >= cfg.Ca:
            continue

        # utility proxy comparison
        util_old = float(beta[a_old, u] - cfg.lambda_I * (beta_sum_ap[a_old] - beta[a_old, u]))
        util_new = float(beta[a_new, u] - cfg.lambda_I * (beta_sum_ap[a_new] - beta[a_new, u]))

        if util_new > util_old:
            A_mat[u, a_old] = 0
            load[a_old] -= 1
            A_mat[u, a_new] = 1
            load[a_new] += 1

    return A_mat

# -----------------------
# RB Scheduling (capacity Q) using interference proxy + pilot contamination penalty
# -----------------------
def rb_matching_capacity(beta: np.ndarray, pilot_id: np.ndarray, A_mat: np.ndarray, p: np.ndarray, cfg: SimConfig):
    """
    Greedy packing of UEs into K RBs (capacity Q).
    若 K*Q < U：只排程最多 K*Q 個 UE，其餘 UE 不排程（S 全 0），更符合現實。

    cost between (u,v):
      base = kappa[u,v] + kappa[v,u]
      pc penalty if same pilot: lambda_pc * base
      total incr cost = lambda_I*base + pc

    Return:
      S [U,K] one-hot for scheduled users; unscheduled users are all-zero row.
    """
    U, K, Q = cfg.n_ue, cfg.K, cfg.Q
    rng = np.random.default_rng(cfg.seed)

    # precompute proxy coupling kappa[u,v] = sum_{a in A_u} beta[a,v]
    kappa = (A_mat @ beta).astype(np.float32)  # [U,U]

    groups = [[] for _ in range(K)]
    load = np.zeros((K,), dtype=np.int32)

    order = np.arange(U)
    rng.shuffle(order)

    # 只排程最多 K*Q 個 UE
    max_sched = min(U, K * Q)
    sched_users = order[:max_sched]
    # unsched_users = order[max_sched:]  # 不需特別用到，S 會保持全 0

    def incr_cost(u, k):
        c = 0.0
        for v in groups[k]:
            base = float(kappa[u, v] + kappa[v, u])
            pc = 0.0
            if pilot_id[u] == pilot_id[v]:
                pc = cfg.lambda_pc * base
            c += cfg.lambda_I * base + pc
        return c

    for u in sched_users:
        u = int(u)
        best_k = None
        best_val = 1e30
        for k in range(K):
            if load[k] >= Q:
                continue
            val = incr_cost(u, k)
            if val < best_val:
                best_val = val
                best_k = k

        # 理論上 best_k 一定找得到，除非 K 或 Q 設定不合理
        if best_k is None:
            # fallback：不排程
            continue

        groups[best_k].append(u)
        load[best_k] += 1

    # build S
    S = np.zeros((U, K), dtype=np.int32)
    for k in range(K):
        for u in groups[k]:
            S[u, k] = 1

    return S

def groups_from_S(S: np.ndarray, cfg: SimConfig):
    return [np.where(S[:, k] == 1)[0].tolist() for k in range(cfg.K)]

# -----------------------
# Local LMMSE combining per AP, per RB, using estimated channels Hhat
# -----------------------
def local_lmmse_combining(Hhat: np.ndarray, A_mat: np.ndarray, S: np.ndarray, p: np.ndarray, cfg: SimConfig):
    """
    v[a,u] = (sum_{v in group} p_v * hhat[a,v]hhat[a,v]^H + sigma^2 I)^-1 hhat[a,u]
    Only compute if A_mat[u,a]=1.
    Return V [A,U,M] complex64
    """
    A, U, M = cfg.n_ap, cfg.n_ue, cfg.M
    sigma2 = cfg.noise_power_lin
    V = np.zeros((A, U, M), dtype=np.complex64)
    I = np.eye(M, dtype=np.complex64)

    groups = groups_from_S(S, cfg)
    for users in groups:
        if len(users) == 0:
            continue
        users = [int(u) for u in users]
        for a in range(A):
            R = (sigma2 * I).copy()
            for v in users:
                hv = Hhat[a, v, :].reshape(M, 1)
                R += (p[v] * (hv @ hv.conj().T)).astype(np.complex64)

            # solve for each u (served by AP a)
            for u in users:
                if A_mat[u, a] == 0:
                    continue
                hu = Hhat[a, u, :].reshape(M, 1)
                V[a, u, :] = np.linalg.solve(R, hu).reshape(M,).astype(np.complex64)

    return V

# -----------------------
# Effective scalar coupling using TRUE channels H (for SINR)
# -----------------------
def effective_coupling(H: np.ndarray, V: np.ndarray, A_mat: np.ndarray, cfg: SimConfig):
    """
    C[u,v] = sum_{a in cluster(u)} v[a,u]^H h[a,v]
    noise_eff[u] = sigma^2 * sum_{a in cluster(u)} ||v[a,u]||^2
    """
    A, U, M = cfg.n_ap, cfg.n_ue, cfg.M
    sigma2 = cfg.noise_power_lin
    C = np.zeros((U, U), dtype=np.complex64)
    n_eff = np.zeros((U,), dtype=np.float32)

    for u in range(U):
        aps = np.where(A_mat[u] == 1)[0]
        if len(aps) == 0:
            continue
        for a in aps:
            vu = V[a, u, :]
            n_eff[u] += float(sigma2 * np.vdot(vu, vu).real)
            # accumulate coupling to all v
            # np.vdot does conj on first arg
            C[u, :] += np.array([np.vdot(vu, H[a, v, :]) for v in range(U)], dtype=np.complex64)

    return C, n_eff

# -----------------------
# WMMSE power control on scalar effective channels (within RB)
# -----------------------
def wmmse_power(C: np.ndarray, n_eff: np.ndarray, S: np.ndarray, p_init: np.ndarray, cfg: SimConfig, iters: int):
    U = cfg.n_ue
    p = p_init.copy().astype(np.float32)
    vtx = np.sqrt(np.maximum(p, 0.0)).astype(np.float32)

    groups = groups_from_S(S, cfg)
    for _ in range(iters):
        for users in groups:
            if len(users) == 0:
                continue
            users = [int(u) for u in users]

            # u_rx update
            u_rx = np.zeros((len(users),), dtype=np.complex64)
            den = np.zeros((len(users),), dtype=np.float32)

            for ii, i in enumerate(users):
                d = float(n_eff[i])
                for j in users:
                    d += float((vtx[j]**2) * (np.abs(C[i, j])**2))
                den[ii] = d
                u_rx[ii] = (C[i, i] * vtx[i]) / (d + 1e-12)

            # w update
            w = np.zeros((len(users),), dtype=np.float32)
            for ii, i in enumerate(users):
                term = float((u_rx[ii].conjugate() * C[i, i] * vtx[i]).real)
                e = 1.0 - 2.0*term + (np.abs(u_rx[ii])**2)*den[ii]
                e = max(e, 1e-6)
                w[ii] = 1.0 / e

            # v update
            for ii, i in enumerate(users):
                num = float(w[ii] * (u_rx[ii].conjugate() * C[i, i]).real)
                d = 1e-12
                for kk, k in enumerate(users):
                    d += float(w[kk] * (np.abs(u_rx[kk])**2) * (np.abs(C[k, i])**2))
                v_new = max(0.0, num / d)
                v_new = min(v_new, math.sqrt(cfg.pmax))
                vtx[i] = v_new
                p[i] = float(min(cfg.pmax, max(0.0, v_new*v_new)))

    return p

# -----------------------
# Rate evaluation (instantaneous SINR + prelog)
# -----------------------
def sum_rate(C: np.ndarray, n_eff: np.ndarray, S: np.ndarray, p: np.ndarray, cfg: SimConfig):
    """
    Sum-rate with prelog, ignoring unscheduled users (rows with S[u,:]=0).
    """
    U = cfg.n_ue
    total = 0.0
    pl = prelog(cfg)

    groups = groups_from_S(S, cfg)

    # 對每個 u 找到其 RB；若未排程則跳過
    for u in range(U):
        if S[u].sum() == 0:
            continue
        k = int(np.argmax(S[u]))
        users = groups[k]

        sig = p[u] * (np.abs(C[u, u])**2)
        interf = 0.0
        for v in users:
            if v == u:
                continue
            interf += p[v] * (np.abs(C[u, v])**2)

        sinr = float(sig / (interf + float(n_eff[u]) + 1e-12))
        total += math.log2(1.0 + sinr)

    return pl * float(total)

# -----------------------
# Teacher AO (the baseline you will publish)
# -----------------------
def teacher_ao(beta, pilot_id, H, Hhat, cfg: SimConfig):
    """
    AO loop:
      - clustering init (Top-L + load)
      repeat:
        - clustering swap improve (heuristic)
        - RB matching (capacity Q, interference+pilot penalty)
        - local LMMSE combining (uses Hhat)
        - effective coupling from TRUE H
        - WMMSE power
    return A_mat, S, p, C, n_eff, SR
    """
    rng = np.random.default_rng(cfg.seed)

    A_mat = init_topL_clustering(beta, cfg)
    p = np.full((cfg.n_ue,), 0.5*cfg.pmax, dtype=np.float32)

    # init schedule
    S = rb_matching_capacity(beta, pilot_id, A_mat, p, cfg)

    for _ in range(cfg.ao_iters):
        A_mat = clustering_swap_improve(beta, A_mat, cfg, rng)
        S = rb_matching_capacity(beta, pilot_id, A_mat, p, cfg)

        V = local_lmmse_combining(Hhat, A_mat, S, p, cfg)
        C, n_eff = effective_coupling(H, V, A_mat, cfg)
        p = wmmse_power(C, n_eff, S, p, cfg, iters=cfg.wmmse_iters)

    # final
    V = local_lmmse_combining(Hhat, A_mat, S, p, cfg)
    C, n_eff = effective_coupling(H, V, A_mat, cfg)
    sr = sum_rate(C, n_eff, S, p, cfg)

    return A_mat.astype(np.int32), S.astype(np.int32), p.astype(np.float32), C, n_eff, sr
