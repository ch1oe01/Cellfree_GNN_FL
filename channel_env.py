# channel_env.py
import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import RayleighBlockFading
from typing import Tuple, Optional, Dict, Any


class SionnaCellFreeUplinkEnv:
    """
    6G-ish Cell-free Uplink Environment (multi-AP cooperation)

    新增重點：固定拓樸 (fixed_topology)
    - fixed_topology=True：同一個 episode / eval run 內，AP/UE 位置固定，只變小尺度 fading
    - 這能讓「C=1 vs C>1 協作增益」更清楚、不會被每 slot 重抽位置沖掉
    """

    def __init__(
        self,
        num_ap: int = 8,
        num_ue: int = 64,
        num_rb: int = 16,
        area_side_m: float = 500.0,
        carrier_freq_hz: float = 3.5e9,
        num_ap_rx_ant: int = 16,
        num_ue_tx_ant: int = 1,
        pathloss_exp: float = 3.2,
        shadowing_std_db: float = 8.0,
        min_dist_m: float = 5.0,
        rb_bw_hz: float = 500.0,
        noise_psd_w_per_hz: float = 1e-20,
        pmax_watt: float = 0.2,
        seed: int = 7,
        debug: bool = False,
        # ---- 6G-ish scheduler knobs ----
        pf_ema_alpha: float = 0.02,
        pf_eps: float = 1e-6,
        # ---- NEW: fixed topology ----
        fixed_topology: bool = True,
    ):
        self.A = int(num_ap)
        self.U = int(num_ue)
        self.K = int(num_rb)
        self.area = float(area_side_m)
        self.fc = float(carrier_freq_hz)

        self.Nr = int(num_ap_rx_ant)
        self.Nt = int(num_ue_tx_ant)

        self.alpha = float(pathloss_exp)
        self.sigma_sh_db = float(shadowing_std_db)
        self.min_dist = float(min_dist_m)

        self.W = float(rb_bw_hz)
        self.N0 = float(noise_psd_w_per_hz)
        self.pmax = float(pmax_watt)

        self.debug = bool(debug)
        self.pf_ema_alpha = float(pf_ema_alpha)
        self.pf_eps = float(pf_eps)

        self.fixed_topology = bool(fixed_topology)

        tf.random.set_seed(int(seed))
        self._np_rng = np.random.default_rng(int(seed))

        # 固定 AP 位置（要每次 run 重抽，可呼叫 resample_ap_positions）
        self.ap_pos = self._uniform_positions(self.A)

        self.rayleigh = RayleighBlockFading(
            num_rx=self.A,
            num_rx_ant=self.Nr,
            num_tx=self.U,
            num_tx_ant=self.Nt
        )

        # PF state
        self.reset_pf_state()

        # ---- NEW: topology cache (for fixed_topology) ----
        self.ue_pos_cache: Optional[tf.Tensor] = None     # [U,2]
        self.d_ap_ue_cache: Optional[tf.Tensor] = None    # [A,U]
        self.serving_ap_cache: Optional[tf.Tensor] = None # [U]

        # 如果 fixed_topology=True，初始化就先建立一個固定拓樸
        if self.fixed_topology:
            self.reset_topology()

        print(f"✅ Env init | Sionna v{sionna.__version__} | A={self.A},U={self.U},K={self.K},Nr={self.Nr} | fixed_topology={self.fixed_topology}")

    # ----------------------------
    # PF state
    # ----------------------------
    def reset_pf_state(self):
        self.avg_rate = tf.ones([self.U], dtype=tf.float32) * tf.cast(1.0, tf.float32)

    # ----------------------------
    # Topology
    # ----------------------------
    def _uniform_positions(self, n: int) -> tf.Tensor:
        return tf.random.uniform([n, 2], 0.0, self.area, dtype=tf.float32)

    def resample_ap_positions(self):
        self.ap_pos = self._uniform_positions(self.A)
        # AP 變了，固定拓樸下也要更新距離
        if self.fixed_topology and self.ue_pos_cache is not None:
            self.reset_topology(ue_pos=self.ue_pos_cache)

    def sample_ue_positions(self) -> tf.Tensor:
        return self._uniform_positions(self.U)

    def build_topology(self, ue_pos: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        return:
          ue_pos: [U,2]
          d: [A,U]
          serving_ap: [U]
        """
        if ue_pos is None:
            ue_pos = self.sample_ue_positions()

        ap_xy = tf.expand_dims(self.ap_pos, 1)          # [A,1,2]
        ue_xy = tf.expand_dims(ue_pos, 0)               # [1,U,2]
        d = tf.norm(ap_xy - ue_xy, axis=-1)             # [A,U]
        d = tf.maximum(d, self.min_dist)
        serving_ap = tf.argmin(d, axis=0, output_type=tf.int32)
        return ue_pos, d, serving_ap

    def reset_topology(self, ue_pos: Optional[tf.Tensor] = None):
        """
        NEW：固定拓樸用
        - ue_pos 不給：就重抽一個 UE 位置
        - 結果會 cache 起來，sample_channel() 會直接用 cache 的距離矩陣
        """
        ue_pos, d, serving_ap = self.build_topology(ue_pos=ue_pos)
        self.ue_pos_cache = ue_pos
        self.d_ap_ue_cache = d
        self.serving_ap_cache = serving_ap

    def get_clusters_nearest(self, d_ap_ue: tf.Tensor, C: int = 3) -> tf.Tensor:
        C = int(C)
        _, idx = tf.math.top_k(-tf.transpose(d_ap_ue), k=C)  # [U,A] -> idx [U,C]
        return tf.cast(idx, tf.int32)

    # ----------------------------
    # Large-scale helpers
    # ----------------------------
    def _friis_pl0_db_at_1m(self) -> tf.Tensor:
        c = 299_792_458.0
        pl0 = 20.0 * tf.math.log(4.0 * 3.141592653589793 * self.fc / c) / tf.math.log(10.0)
        return tf.cast(pl0, tf.float32)

    def noise_power_per_rb(self) -> tf.Tensor:
        return tf.cast(self.N0 * self.W, tf.float32)

    # ----------------------------
    # Sionna reshape helper
    # ----------------------------
    def _sionna_to_h_baNruk(self, h_raw: tf.Tensor) -> tf.Tensor:
        if self.debug:
            tf.print("DEBUG raw h_sionna shape:", tf.shape(h_raw))

        h_s = h_raw

        # squeeze tx_ant (Nt=1) at axis=4
        if h_s.shape.rank is not None and h_s.shape.rank >= 5 and h_s.shape[4] == 1:
            h_s = tf.squeeze(h_s, axis=4)

        # handle rank=6 with possible (paths, K) ordering
        if h_s.shape.rank == 6:
            if h_s.shape[-1] == self.K:
                if h_s.shape[-2] == 1:
                    h_s = tf.squeeze(h_s, axis=-2)
            else:
                if h_s.shape[-1] == 1:
                    h_s = tf.squeeze(h_s, axis=-1)

        if self.debug:
            tf.print("DEBUG h_s after squeeze shape:", tf.shape(h_s))

        # still rank=6 => take first path
        if h_s.shape.rank == 6:
            if h_s.shape[-1] == self.K:
                h_s = h_s[..., 0, :]
            else:
                h_s = h_s[..., 0]

        if self.debug:
            tf.print("DEBUG final h_s shape:", tf.shape(h_s))

        return tf.cast(h_s, tf.complex64)

    # ----------------------------
    # Channel sampling (FIXED topology supported)
    # ----------------------------
    def sample_channel(self, batch_size: int = 1):
        """
        return:
          h: [B,A,Nr,U,K] complex64
          g: [B,A,U,K] float32
          d: [A,U] float32
        """
        B = int(batch_size)

        # NEW：若 fixed_topology=True，就用 cache 的 d；否則每次重抽拓樸
        if self.fixed_topology:
            if self.d_ap_ue_cache is None:
                self.reset_topology()
            d_ap_ue = self.d_ap_ue_cache
        else:
            _, d_ap_ue, _ = self.build_topology()

        # Large-scale fading
        pl0 = self._friis_pl0_db_at_1m()
        shadow = tf.random.normal([self.A, self.U], 0.0, self.sigma_sh_db, dtype=tf.float32)
        pl_db = pl0 + 10.0 * self.alpha * (tf.math.log(d_ap_ue) / tf.math.log(10.0)) + shadow
        beta = tf.pow(10.0, -pl_db / 10.0)  # [A,U]

        # Small-scale fading from Sionna (time_steps = K)
        h_tuple = self.rayleigh(B, num_time_steps=self.K)
        h_raw = h_tuple[0]
        h_s = self._sionna_to_h_baNruk(h_raw)  # [B,A,Nr,U,K]

        # Scale by sqrt(beta)
        beta_b = tf.expand_dims(beta, axis=0)      # [1,A,U]
        beta_b = tf.expand_dims(beta_b, axis=2)    # [1,A,1,U]
        beta_b = tf.expand_dims(beta_b, axis=-1)   # [1,A,1,U,1]
        h = tf.cast(tf.sqrt(beta_b), tf.complex64) * tf.cast(h_s, tf.complex64)

        g = tf.reduce_mean(tf.abs(h) ** 2, axis=2)  # [B,A,U,K]
        g = tf.cast(g, tf.float32)

        return h, g, tf.cast(d_ap_ue, tf.float32)

    # ----------------------------
    # constraints
    # ----------------------------
    @staticmethod
    def validate_schedule_x(x_uk: tf.Tensor, L: int):
        x = tf.cast(x_uk, tf.float32)
        ue_sum = tf.reduce_sum(x, axis=2)  # [B,U]
        rb_sum = tf.reduce_sum(x, axis=1)  # [B,K]
        tf.debugging.assert_less_equal(ue_sum, 1.0 + 1e-6, message="UE 使用超過 1 個 RB")
        tf.debugging.assert_less_equal(rb_sum, float(L) + 1e-6, message="RB 同時 UE 超過 L")

    # ---- greedy matching helper (same as yours) ----
    def _greedy_matching_one_rb(self, score_buk: np.ndarray, L: int, force_all: bool = False) -> np.ndarray:
        U, K = score_buk.shape
        cap_total = K * L
        if force_all and cap_total < U:
            raise ValueError(f"force_all=True 但容量不足：K*L={cap_total} < U={U}")

        x = np.zeros((U, K), dtype=np.float32)
        assigned = np.full(U, -1, dtype=np.int32)
        rb_cnt = np.zeros(K, dtype=np.int32)

        pairs = [(u, k, float(score_buk[u, k])) for u in range(U) for k in range(K)]
        pairs.sort(key=lambda t: t[2], reverse=True)

        for u, k, _ in pairs:
            if assigned[u] != -1:
                continue
            if rb_cnt[k] >= L:
                continue
            assigned[u] = k
            rb_cnt[k] += 1
            x[u, k] = 1.0
            if rb_cnt.sum() >= cap_total:
                break

        if force_all:
            un = np.where(assigned == -1)[0]
            for u in un:
                avail = np.where(rb_cnt < L)[0]
                k_best = avail[np.argmax(score_buk[u, avail])]
                assigned[u] = k_best
                rb_cnt[k_best] += 1
                x[u, k_best] = 1.0

        return x

    # ----------------------------
    # schedulers: random / gain / pf (same behavior as before)
    # ----------------------------
    @staticmethod
    def schedule_random_one_rb(B: int, U: int, K: int, L: int, seed: int = 123, force_all: bool = False) -> tf.Tensor:
        rng = np.random.default_rng(seed)
        cap_total = K * L
        if force_all and cap_total < U:
            raise ValueError(f"force_all=True 但容量不足：K*L={cap_total} < U={U}")

        x = np.zeros((B, U, K), dtype=np.float32)
        for b in range(B):
            ue_perm = rng.permutation(U)
            ue_sel = ue_perm if force_all else ue_perm[:min(U, cap_total)]
            ptr = 0
            for k in range(K):
                take = min(L, len(ue_sel) - ptr)
                if take <= 0:
                    break
                users = ue_sel[ptr:ptr + take]
                x[b, users, k] = 1.0
                ptr += take
        return tf.convert_to_tensor(x, dtype=tf.float32)

    def schedule_gain_matching_one_rb(self, h: tf.Tensor, ap_idx_uc: tf.Tensor, L: int, force_all: bool = False) -> tf.Tensor:
        B = int(h.shape[0])
        U = int(h.shape[3])
        K = int(h.shape[4])
        L = int(L)

        mask_ap = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=self.A, dtype=tf.float32), axis=1)  # [U,A]
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)                                    # [U,R]

        h_flat = tf.reshape(h, [B, self.A * self.Nr, U, K])     # [B,R,U,K]
        H = tf.transpose(h_flat, [0, 2, 1, 3])                  # [B,U,R,K]

        s = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)   # [B,U,K]
        s = tf.cast(s, tf.float32)

        s_np = s.numpy()
        x_np = np.zeros((B, U, K), dtype=np.float32)
        for b in range(B):
            x_np[b] = self._greedy_matching_one_rb(s_np[b], L=L, force_all=force_all)
        return tf.convert_to_tensor(x_np, dtype=tf.float32)

    def compute_sum_rate_mrc_clustered(self, h: tf.Tensor, x_uk: tf.Tensor, ap_idx_uc: tf.Tensor, p_uk: Optional[tf.Tensor] = None):
        x = tf.cast(x_uk, tf.float32)
        if p_uk is None:
            p_uk = tf.ones_like(x) * tf.cast(self.pmax, tf.float32)
        p_act = tf.cast(p_uk, tf.float32) * x  # [B,U,K]

        B = tf.shape(h)[0]
        U = tf.shape(h)[3]
        K = tf.shape(h)[4]

        mask_ap = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=self.A, dtype=tf.float32), axis=1)  # [U,A]
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)                                    # [U,R]

        h_flat = tf.reshape(h, [B, self.A * self.Nr, U, K])   # [B,R,U,K]
        H = tf.transpose(h_flat, [0, 2, 1, 3])                # [B,U,R,K]

        s = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)  # [B,U,K]
        s = tf.cast(s, tf.float32)

        mask_r_c = tf.cast(mask_r, H.dtype)
        ip = tf.einsum('ur,burk,bvrk->buvk', mask_r_c, tf.math.conj(H), H)  # [B,U,V,K]
        corr_pow = tf.abs(ip) ** 2

        total = tf.reduce_sum(tf.expand_dims(p_act, axis=1) * tf.cast(corr_pow, tf.float32), axis=2)  # [B,U,K]
        signal = p_act * (s ** 2)
        interf = total - signal

        noise = self.noise_power_per_rb()
        noise_term = noise * s

        sinr = signal / (interf + noise_term + 1e-12)
        rate = self.W * (tf.math.log(1.0 + sinr) / tf.math.log(2.0))
        sum_rate = tf.reduce_sum(rate)
        return sum_rate, rate, sinr

    def schedule_pf_one_rb(self, h: tf.Tensor, ap_idx_uc: tf.Tensor, L: int, force_all: bool = False) -> tf.Tensor:
        B = int(h.shape[0])
        U = int(h.shape[3])
        K = int(h.shape[4])
        L = int(L)

        mask_ap = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=self.A, dtype=tf.float32), axis=1)  # [U,A]
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)                                    # [U,R]

        h_flat = tf.reshape(h, [B, self.A * self.Nr, U, K])     # [B,R,U,K]
        H = tf.transpose(h_flat, [0, 2, 1, 3])                  # [B,U,R,K]

        s = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)   # [B,U,K]
        s = tf.cast(s, tf.float32)

        noise = self.noise_power_per_rb()
        sinr0 = (tf.cast(self.pmax, tf.float32) * s) / (noise + 1e-12)
        inst_rate0 = self.W * (tf.math.log(1.0 + sinr0) / tf.math.log(2.0))

        avg = tf.reshape(self.avg_rate + self.pf_eps, [1, U, 1])
        score = inst_rate0 / avg

        score_np = score.numpy()
        x_np = np.zeros((B, U, K), dtype=np.float32)
        for b in range(B):
            x_np[b] = self._greedy_matching_one_rb(score_np[b], L=L, force_all=force_all)
        return tf.convert_to_tensor(x_np, dtype=tf.float32)

    def pf_update(self, rate_buk: tf.Tensor):
        r_u = tf.reduce_mean(tf.reduce_sum(rate_buk, axis=2), axis=0)  # [U]
        self.avg_rate = (1.0 - self.pf_ema_alpha) * self.avg_rate + self.pf_ema_alpha * tf.cast(r_u, tf.float32)

    def step(self, batch_size: int, C: int, L: int, scheduler: str = "pf", force_all: bool = False, p_uk: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        h, g, d = self.sample_channel(batch_size=batch_size)
        ap_idx = self.get_clusters_nearest(d, C=C)

        if scheduler == "random":
            x = self.schedule_random_one_rb(batch_size, self.U, self.K, L, seed=int(self._np_rng.integers(1e9)), force_all=force_all)
        elif scheduler == "gain":
            x = self.schedule_gain_matching_one_rb(h, ap_idx, L=L, force_all=force_all)
        elif scheduler == "pf":
            x = self.schedule_pf_one_rb(h, ap_idx, L=L, force_all=force_all)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        self.validate_schedule_x(x, L)
        sum_rate, rate, sinr = self.compute_sum_rate_mrc_clustered(h, x, ap_idx, p_uk=p_uk)

        if scheduler == "pf":
            self.pf_update(rate)

        return {"h": h, "g": g, "d": d, "ap_idx": ap_idx, "x": x, "sum_rate": sum_rate, "rate": rate, "sinr": sinr, "avg_rate": self.avg_rate}


if __name__ == "__main__":
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    env = SionnaCellFreeUplinkEnv(num_ap=8, num_ue=64, num_rb=16, debug=False, fixed_topology=True)
    # 固定拓樸下跑一個 slot
    out = env.step(batch_size=1, C=3, L=1, scheduler="pf", force_all=False)
    print("sum_rate:", float(out["sum_rate"].numpy()))
