# channel_env.py
import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import RayleighBlockFading
from typing import Tuple, Optional, Dict, Any


class SionnaCellFreeUplinkEnv:
    """
    6G-ish Cell-free Uplink Environment (multi-AP cooperation)
    ----------------------------------------------------------
    - 小尺度：Sionna RayleighBlockFading (block fading per RB)
    - 大尺度：Friis@1m + log-distance pathloss + shadowing
    - 協作：UE->最近 C 個 AP 組成 cluster，採 MRC joint reception
    - 排程限制：
        * 每 UE 每 slot 最多 1 個 RB (sum_k x[u,k] <= 1)
        * 每 RB 最多 L 個 UE (sum_u x[u,k] <= L) 允許 overloading
    - 6G 真實感：
        * full-buffer：每個 UE 都有資料（都想傳），但不一定每 slot 都排到
        * PF fairness：依長期平均速率做比例公平排程
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
        pf_ema_alpha: float = 0.02,   # PF 平均速率 EMA 更新係數（小一點更平滑）
        pf_eps: float = 1e-6,         # 避免除以 0
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

        # random seed
        tf.random.set_seed(int(seed))
        self._np_rng = np.random.default_rng(int(seed))

        # 固定 AP 位置（若你要每 episode 重抽，可呼叫 resample_ap_positions）
        self.ap_pos = self._uniform_positions(self.A)

        # Sionna Rayleigh block fading：time_steps 用來當 RB index
        self.rayleigh = RayleighBlockFading(
            num_rx=self.A,
            num_rx_ant=self.Nr,
            num_tx=self.U,
            num_tx_ant=self.Nt
        )

        # PF 狀態：長期平均速率 (bps)，reset 時給一個小正數避免 0
        self.reset_pf_state()

        print(f"✅ Env init | Sionna v{sionna.__version__} | A={self.A},U={self.U},K={self.K},Nr={self.Nr}")

    # ----------------------------
    # State reset (PF)
    # ----------------------------
    def reset_pf_state(self):
        # avg_rate[u] : float32, bps
        self.avg_rate = tf.ones([self.U], dtype=tf.float32) * tf.cast(1.0, tf.float32)

    # ----------------------------
    # Topology & Large-scale fading
    # ----------------------------
    def _uniform_positions(self, n: int) -> tf.Tensor:
        return tf.random.uniform([n, 2], 0.0, self.area, dtype=tf.float32)

    def resample_ap_positions(self):
        self.ap_pos = self._uniform_positions(self.A)

    def sample_ue_positions(self) -> tf.Tensor:
        return self._uniform_positions(self.U)

    def _friis_pl0_db_at_1m(self) -> tf.Tensor:
        # PL0(dB) = 20 log10(4*pi*fc/c)
        c = 299_792_458.0
        pl0 = 20.0 * tf.math.log(4.0 * 3.141592653589793 * self.fc / c) / tf.math.log(10.0)
        return tf.cast(pl0, tf.float32)

    def noise_power_per_rb(self) -> tf.Tensor:
        return tf.cast(self.N0 * self.W, tf.float32)

    def build_topology(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        return:
          ue_pos: [U,2]
          d: [A,U]
          serving_ap: [U]
        """
        ue_pos = self.sample_ue_positions()             # [U,2]
        ap_xy = tf.expand_dims(self.ap_pos, 1)          # [A,1,2]
        ue_xy = tf.expand_dims(ue_pos, 0)               # [1,U,2]
        d = tf.norm(ap_xy - ue_xy, axis=-1)             # [A,U]
        d = tf.maximum(d, self.min_dist)
        serving_ap = tf.argmin(d, axis=0, output_type=tf.int32)
        return ue_pos, d, serving_ap

    def get_clusters_nearest(self, d_ap_ue: tf.Tensor, C: int = 3) -> tf.Tensor:
        """
        UE 的協作 AP cluster：最近 C 個 AP
        d_ap_ue: [A,U]
        return ap_idx_uc: [U,C]
        """
        C = int(C)
        _, idx = tf.math.top_k(-tf.transpose(d_ap_ue), k=C)  # input [U,A] -> idx [U,C]
        return tf.cast(idx, tf.int32)

    # ----------------------------
    # Sionna output reshape helper
    # ----------------------------
    def _sionna_to_h_baNruk(self, h_raw: tf.Tensor) -> tf.Tensor:
        """
        把 RayleighBlockFading 輸出整理成 [B, A, Nr, U, K] complex64

        常見 raw shape：
          [B, A, Nr, U, Nt(=1), paths(=1), K]
        或：
          [B, A, Nr, U, Nt(=1), K, paths(=1)]
        """
        if self.debug:
            tf.print("DEBUG raw h_sionna shape:", tf.shape(h_raw))

        h_s = h_raw

        # squeeze tx_ant (Nt=1) 通常在 axis=4
        if h_s.shape.rank is not None and h_s.shape.rank >= 5 and h_s.shape[4] == 1:
            h_s = tf.squeeze(h_s, axis=4)

        # 這時候常見是 rank=6: [B,A,Nr,U,?,?]
        if h_s.shape.rank == 6:
            # 若最後一維是 K，代表倒數第二維應是 paths(=1) -> squeeze(-2)
            if h_s.shape[-1] == self.K:
                if h_s.shape[-2] == 1:
                    h_s = tf.squeeze(h_s, axis=-2)
            else:
                # 否則可能是 [..., K, 1] -> squeeze(-1)
                if h_s.shape[-1] == 1:
                    h_s = tf.squeeze(h_s, axis=-1)

        if self.debug:
            tf.print("DEBUG h_s after squeeze shape:", tf.shape(h_s))

        # 若仍是 6 維（paths > 1），保守取第一個 path
        if h_s.shape.rank == 6:
            if h_s.shape[-1] == self.K:
                # [B,A,Nr,U,P,K]
                h_s = h_s[..., 0, :]
            else:
                # [B,A,Nr,U,K,P]
                h_s = h_s[..., 0]

        if self.debug:
            tf.print("DEBUG final h_s shape:", tf.shape(h_s))

        return tf.cast(h_s, tf.complex64)

    # ----------------------------
    # Channel sampling
    # ----------------------------
    def sample_channel(self, batch_size: int = 1):
        """
        return:
          h: [B,A,Nr,U,K] complex64
          g: [B,A,U,K] float32
          d: [A,U] float32
        """
        B = int(batch_size)
        _, d_ap_ue, _ = self.build_topology()  # [A,U]

        # Large-scale fading
        pl0 = self._friis_pl0_db_at_1m()
        shadow = tf.random.normal([self.A, self.U], 0.0, self.sigma_sh_db, dtype=tf.float32)
        pl_db = pl0 + 10.0 * self.alpha * (tf.math.log(d_ap_ue) / tf.math.log(10.0)) + shadow
        beta = tf.pow(10.0, -pl_db / 10.0)  # power gain [A,U]

        # Small-scale fading from Sionna (time_steps = K)
        h_tuple = self.rayleigh(B, num_time_steps=self.K)
        h_raw = h_tuple[0]
        h_s = self._sionna_to_h_baNruk(h_raw)  # [B,A,Nr,U,K]

        # Scale by sqrt(beta)
        beta_b = tf.expand_dims(beta, axis=0)      # [1,A,U]
        beta_b = tf.expand_dims(beta_b, axis=2)    # [1,A,1,U]
        beta_b = tf.expand_dims(beta_b, axis=-1)   # [1,A,1,U,1]
        h = tf.cast(tf.sqrt(beta_b), tf.complex64) * tf.cast(h_s, tf.complex64)

        # Power gain averaged over antennas
        g = tf.reduce_mean(tf.abs(h) ** 2, axis=2)  # [B,A,U,K]
        g = tf.cast(g, tf.float32)

        return h, g, d_ap_ue

    # ----------------------------
    # Schedule constraints helpers
    # ----------------------------
    @staticmethod
    def validate_schedule_x(x_uk: tf.Tensor, L: int):
        """
        檢查：
        (1) 每個 UE 最多 1 個 RB：sum_k x[b,u,k] <= 1
        (2) 每個 RB 最多 L 個 UE：sum_u x[b,u,k] <= L
        """
        x = tf.cast(x_uk, tf.float32)
        ue_sum = tf.reduce_sum(x, axis=2)  # [B,U]
        rb_sum = tf.reduce_sum(x, axis=1)  # [B,K]
        tf.debugging.assert_less_equal(ue_sum, 1.0 + 1e-6, message="UE 使用超過 1 個 RB")
        tf.debugging.assert_less_equal(rb_sum, float(L) + 1e-6, message="RB 同時 UE 超過 L")

    def _greedy_matching_one_rb(
        self,
        score_buk: np.ndarray,
        L: int,
        force_all: bool = False
    ) -> np.ndarray:
        """
        score_buk: [U,K] numpy
        return x: [U,K] numpy 0/1
        限制：每 UE <=1 RB；每 RB <=L UE
        force_all=True：要求每個 UE 都要被分配到 1 個 RB（需 K*L >= U，否則丟錯）
        """
        U, K = score_buk.shape
        cap_total = K * L

        if force_all and cap_total < U:
            raise ValueError(f"force_all=True 但容量不足：K*L={cap_total} < U={U}")

        x = np.zeros((U, K), dtype=np.float32)
        assigned = np.full(U, -1, dtype=np.int32)
        rb_cnt = np.zeros(K, dtype=np.int32)

        # 排序所有 (u,k)
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

        # 如果 force_all，還有沒分到的 UE，就用「剩餘容量」補齊（較像系統保底服務）
        if force_all:
            un = np.where(assigned == -1)[0]
            if len(un) > 0:
                for u in un:
                    # 找一個還有容量的 RB（選 score 最大的可用 RB）
                    avail = np.where(rb_cnt < L)[0]
                    # 理論上一定存在
                    k_best = avail[np.argmax(score_buk[u, avail])]
                    assigned[u] = k_best
                    rb_cnt[k_best] += 1
                    x[u, k_best] = 1.0

        return x

    # ----------------------------
    # Baseline schedulers (6G-ish)
    # ----------------------------
    @staticmethod
    def schedule_random_one_rb(B: int, U: int, K: int, L: int, seed: int = 123, force_all: bool = False) -> tf.Tensor:
        """
        Random 但符合限制：
        - 每 UE <=1 RB
        - 每 RB <=L UE
        force_all=True：每個 UE 一定要分到 RB（需 K*L >= U）
        """
        rng = np.random.default_rng(seed)
        cap_total = K * L
        if force_all and cap_total < U:
            raise ValueError(f"force_all=True 但容量不足：K*L={cap_total} < U={U}")

        x = np.zeros((B, U, K), dtype=np.float32)
        for b in range(B):
            ue_perm = rng.permutation(U)

            if force_all:
                ue_sel = ue_perm  # 全部都要排上去
            else:
                ue_sel = ue_perm[:min(U, cap_total)]  # 容量不足則剩下不排（更符合現實）

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
        """
        以 cluster 上 ||h||^2 當 instant score，做 greedy matching（更符合你的限制）。
        h: [B,A,Nr,U,K]
        ap_idx_uc: [U,C]
        return x: [B,U,K]
        """
        B = int(h.shape[0])
        U = int(h.shape[3])
        K = int(h.shape[4])
        L = int(L)

        # mask_ap[U,A] -> mask_r[U,R]
        mask_ap = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=self.A, dtype=tf.float32), axis=1)  # [U,A]
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)                                    # [U,R]

        # H: [B,U,R,K]
        h_flat = tf.reshape(h, [B, self.A * self.Nr, U, K])     # [B,R,U,K]
        H = tf.transpose(h_flat, [0, 2, 1, 3])                  # [B,U,R,K]

        # s[b,u,k] = sum_{r in cluster} |H|^2
        s = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)   # [B,U,K]
        s = tf.cast(s, tf.float32)

        s_np = s.numpy()
        x_np = np.zeros((B, U, K), dtype=np.float32)
        for b in range(B):
            x_np[b] = self._greedy_matching_one_rb(s_np[b], L=L, force_all=force_all)

        return tf.convert_to_tensor(x_np, dtype=tf.float32)

    # ----------------------------
    # Clustered MRC Joint Reception sum-rate
    # ----------------------------
    def compute_sum_rate_mrc_clustered(
        self,
        h: tf.Tensor,
        x_uk: tf.Tensor,
        ap_idx_uc: tf.Tensor,
        p_uk: Optional[tf.Tensor] = None
    ):
        """
        Clustered MRC JR:
        h: [B,A,Nr,U,K] complex64
        x_uk: [B,U,K] {0,1}
        ap_idx_uc: [U,C]
        p_uk: [B,U,K] power (W). None => pmax

        return:
          sum_rate (scalar),
          rate [B,U,K],
          sinr [B,U,K]
        """
        x = tf.cast(x_uk, tf.float32)
        if p_uk is None:
            p_uk = tf.ones_like(x) * tf.cast(self.pmax, tf.float32)
        p_act = tf.cast(p_uk, tf.float32) * x  # [B,U,K]

        B = tf.shape(h)[0]
        U = tf.shape(h)[3]
        K = tf.shape(h)[4]

        # cluster mask
        mask_ap = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=self.A, dtype=tf.float32), axis=1)  # [U,A]
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)                                    # [U,R]

        # H: [B,U,R,K]
        h_flat = tf.reshape(h, [B, self.A * self.Nr, U, K])   # [B,R,U,K]
        H = tf.transpose(h_flat, [0, 2, 1, 3])                # [B,U,R,K] complex64

        # s = ||h_u||^2 over cluster (real)
        s = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)  # [B,U,K]
        s = tf.cast(s, tf.float32)

        # einsum dtype 必須一致：mask cast 成 complex
        mask_r_c = tf.cast(mask_r, H.dtype)

        # ip(u,v) = h_u^H h_v over cluster (complex)
        ip = tf.einsum('ur,burk,bvrk->buvk', mask_r_c, tf.math.conj(H), H)  # [B,U,V,K]
        corr_pow = tf.abs(ip) ** 2                                          # real

        # total[b,u,k] = sum_v p[v,k] * |ip(u,v)|^2
        total = tf.reduce_sum(
            tf.expand_dims(p_act, axis=1) * tf.cast(corr_pow, tf.float32),
            axis=2
        )  # [B,U,K]

        signal = p_act * (s ** 2)
        interf = total - signal

        noise = self.noise_power_per_rb()
        noise_term = noise * s

        sinr = signal / (interf + noise_term + 1e-12)
        rate = self.W * (tf.math.log(1.0 + sinr) / tf.math.log(2.0))  # bps
        sum_rate = tf.reduce_sum(rate)

        return sum_rate, rate, sinr

    # ----------------------------
    # PF scheduler (6G-ish)
    # ----------------------------
    def schedule_pf_one_rb(
        self,
        h: tf.Tensor,
        ap_idx_uc: tf.Tensor,
        L: int,
        force_all: bool = False
    ) -> tf.Tensor:
        """
        Proportional Fair scheduling (一 UE 一 RB + RB overloading <= L)
        - 用「假設無干擾的瞬時速率」當作 PF 的 instantaneous 指標（便於排程）
        - 再用你真正的 compute_sum_rate_mrc_clustered 去算真實干擾後的 rate 來更新 avg_rate
        """
        B = int(h.shape[0])
        U = int(h.shape[3])
        K = int(h.shape[4])
        L = int(L)

        # 先用 cluster gain s[b,u,k] 做「無干擾近似」instant rate proxy
        mask_ap = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=self.A, dtype=tf.float32), axis=1)  # [U,A]
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)                                    # [U,R]

        h_flat = tf.reshape(h, [B, self.A * self.Nr, U, K])     # [B,R,U,K]
        H = tf.transpose(h_flat, [0, 2, 1, 3])                  # [B,U,R,K]

        s = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)   # [B,U,K]
        s = tf.cast(s, tf.float32)

        # 無干擾近似 SINR ≈ pmax * s / (noise)
        noise = self.noise_power_per_rb()  # scalar
        sinr0 = (tf.cast(self.pmax, tf.float32) * s) / (noise + 1e-12)
        inst_rate0 = self.W * (tf.math.log(1.0 + sinr0) / tf.math.log(2.0))  # [B,U,K]

        # PF score = inst / avg
        avg = tf.reshape(self.avg_rate + self.pf_eps, [1, U, 1])  # [1,U,1]
        score = inst_rate0 / avg                                   # [B,U,K]

        score_np = score.numpy()
        x_np = np.zeros((B, U, K), dtype=np.float32)
        for b in range(B):
            x_np[b] = self._greedy_matching_one_rb(score_np[b], L=L, force_all=force_all)

        return tf.convert_to_tensor(x_np, dtype=tf.float32)

    def pf_update(self, rate_buk: tf.Tensor):
        """
        用真實干擾後的 rate 來更新 PF 的 avg_rate
        rate_buk: [B,U,K] bps
        我們先把每個 UE 在該 slot 的總速率加總：r_u = sum_k rate[b,u,k]
        再用 EMA 更新 avg_rate。
        """
        r_u = tf.reduce_mean(tf.reduce_sum(rate_buk, axis=2), axis=0)  # [U] (對 batch 平均)
        self.avg_rate = (1.0 - self.pf_ema_alpha) * self.avg_rate + self.pf_ema_alpha * tf.cast(r_u, tf.float32)

    # ----------------------------
    # One-step interface (推薦用這個跑 slot)
    # ----------------------------
    def step(
        self,
        batch_size: int,
        C: int,
        L: int,
        scheduler: str = "pf",
        force_all: bool = False,
        p_uk: Optional[tf.Tensor] = None
    ) -> Dict[str, Any]:
        """
        跑一個 slot：
        1) sample channel
        2) cluster
        3) schedule x_uk
        4) compute sum-rate
        5) 若 scheduler=pf，更新 avg_rate

        scheduler: "pf" | "gain" | "random"
        force_all: True 則要求每 UE 都要被排到（必須 K*L >= U）
        """
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

        # constraints check (可先開著，等穩了再關)
        self.validate_schedule_x(x, L)

        sum_rate, rate, sinr = self.compute_sum_rate_mrc_clustered(h, x, ap_idx, p_uk=p_uk)

        if scheduler == "pf":
            self.pf_update(rate)

        return {
            "h": h, "g": g, "d": d,
            "ap_idx": ap_idx,
            "x": x,
            "sum_rate": sum_rate,
            "rate": rate,
            "sinr": sinr,
            "avg_rate": self.avg_rate,
        }


if __name__ == "__main__":
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    env = SionnaCellFreeUplinkEnv(num_ap=8, num_ue=64, num_rb=16, debug=False)

    # 跑一個 slot
    out = env.step(batch_size=1, C=3, L=4, scheduler="pf", force_all=False)

    # 檢查排程限制
    env.validate_schedule_x(out["x"], L=4)

    # 印出關鍵結果
    print("x shape:", out["x"].shape)
    print("UE max RB used:", tf.reduce_max(tf.reduce_sum(out["x"], axis=2)).numpy())  # <=1
    print("RB max users:", tf.reduce_max(tf.reduce_sum(out["x"], axis=1)).numpy())   # <=L
    print("sum_rate(bps):", float(out["sum_rate"].numpy()))
    print("avg_rate mean:", float(tf.reduce_mean(out["avg_rate"]).numpy()))
