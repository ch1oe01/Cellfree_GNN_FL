# channel_env_v2.py
import tensorflow as tf  
import numpy as np       # 處理 Hungarian / 統計 / array 操作
import sionna            # 提供 3GPP TR 38.901 通道模型

from sionna.channel import RayleighBlockFading
from sionna.channel.tr38901 import UMi, Antenna, AntennaArray
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from scipy.optimize import linear_sum_assignment   # Hungarian algorithm（解 assignment 最佳化）
from typing import Dict, Any, Tuple                # 型別註解 (讓程式可讀性更好)


class SionnaCellFreeUplinkEnvV2:

    def __init__(
        self,
        # -------------------------
        # Topology (UMi)
        # -------------------------
        num_ap: int = 32,        # A
        num_ue: int = 16,        # U 
        num_rb: int = 16,        # K 
        area_side_m: float = 200.0,
        deployment: str = "street",    # "street" (推薦) or "uniform"
        street_block_m: float = 50.0,  # 街廓大小（粗略抽象街谷）
        ap_height_m: float = 10.0,     # UMi BS height
        ue_height_m: float = 1.5,      # UE height

        # -------------------------
        # Carrier / waveform (6G-ish)
        # -------------------------
        carrier_freq_hz: float = 7.0e9,  # 6G early FR3-ish (想用 3.5e9 直接改參數)
        rb_bw_hz: float = 180e3,         # 保留你原本的「每個頻域切片」寬度

        # -------------------------
        # Antennas
        # -------------------------
        num_ap_rx_ant: int = 16,
        num_ue_tx_ant: int = 1,

        # -------------------------
        # Noise / power (recommended)
        # -------------------------
        noise_figure_db: float = 7.0,   # 用 NF 算 N0，系統模擬常用 5~9 dB
        pmax_watt: float = 0.2,         # 23 dBm，UE 常見上行功率等級

        # -------------------------
        # UMi environment realism knobs
        # -------------------------
        indoor_prob: float = 0.2,       # 把 UE 一部分標成 indoor（O2I），讓路損更真實
        ue_speed_mps: float = 1.0,      # 走路速度（可設 0，這版不會壞）

        # -------------------------
        # PF / misc
        # -------------------------
        seed: int = 7,
        debug: bool = False,
        pf_ema_alpha: float = 0.02,
        pf_eps: float = 1e-6,           # PF 的 score = r0 / avg_rate，如果 avg_rate 為 0 會爆炸 → eps 防呆必需

        # -------------------------
        # Channel selection
        # -------------------------
        enable_3gpp: bool = True,
        o2i_model: str = "low",         # 要純 outdoor 可以 indoor_prob=0 + o2i_model="none"
    ):
        # ---- store params ----
        self.A = int(num_ap)
        self.U = int(num_ue)
        self.K = int(num_rb)

        self.area = float(area_side_m)
        self.deployment = str(deployment)
        self.street_block_m = float(street_block_m)

        self.fc = float(carrier_freq_hz)
        self.W = float(rb_bw_hz)

        self.Nr = int(num_ap_rx_ant)
        self.Nt = int(num_ue_tx_ant)

        self.ap_height_m = float(ap_height_m)
        self.ue_height_m = float(ue_height_m)

        self.indoor_prob = float(indoor_prob)
        self.ue_speed_mps = float(ue_speed_mps)

        self.pmax = float(pmax_watt)

        self.debug = bool(debug)
        self.pf_ema_alpha = float(pf_ema_alpha)
        self.pf_eps = float(pf_eps)

        self.enable_3gpp = bool(enable_3gpp)

        # TF 和 NumPy 各自設 seed，跑 baseline 才能重現結果
        tf.random.set_seed(int(seed))
        self._np_rng = np.random.default_rng(int(seed))

        # N0 計算，標準的 noise modeling，改 fc、改 RB BW、改系統配置時 noise 都能自洽
        # Physical N0 (W/Hz): kT * NF
        # kT at 290K ≈ 4.0e-21 W/Hz (-174 dBm/Hz)
        kT = 1.38064852e-23 * 290.0                                      # kT 是熱雜訊功率頻譜密度（W/Hz）
        self.N0 = float(kT * (10.0 ** (float(noise_figure_db) / 10.0)))  # 再乘上 10^(NF/10) 把 receiver noise figure 納入

        # ----------------------------
        # AP positions (street)，UMi street canyon 的最大特色就是「點不是在平原亂撒」，而是沿街道分布
        # ----------------------------
        if self.deployment == "street":
            self.ap_pos_2d = self._street_positions(self.A, self.area, self.street_block_m)
        else:
            self.ap_pos_2d = self._uniform_positions(self.A)

        # TR 38.901 的 pathloss/LOS 模型會用高度；把 (x,y) 補上 z=10m
        self.ap_pos_3d = tf.concat([self.ap_pos_2d, tf.fill([self.A, 1], self.ap_height_m)],axis=1)

        # ----------------------------
        # Channel Model
        # ----------------------------
        if self.enable_3gpp:
            print(f"🚀 Initializing 3GPP UMi Channel (Sionna v{sionna.__version__})...")

            # UE Array (system-level common abstraction)
            if self.Nt == 1:
                ut_array = Antenna(
                    polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=self.fc
                )
            else:
                ut_array = AntennaArray(
                    num_rows=1,
                    num_cols=self.Nt,
                    polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=self.fc
                )

            # BS Array (38.901 pattern)，把 Nr 拆成接近方形的 rows×cols
            rows = int(np.sqrt(self.Nr))
            rows = max(rows, 1)
            cols = int(np.ceil(self.Nr / rows))
            cols = max(cols, 1)

            # AP/BS 用 38.901 pattern（有方向性）
            bs_array = AntennaArray(
                num_rows=rows,
                num_cols=cols,
                polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=self.fc
            )

            # 建立 UMi 通道模型，包含 pathloss、shadow fading、O2I
            self.channel_model_3gpp = UMi(
                carrier_frequency=self.fc,
                o2i_model=o2i_model,
                ut_array=ut_array,
                bs_array=bs_array,
                direction="uplink",
                enable_pathloss=True,
                enable_shadow_fading=True
            )

            # 產生 K 個頻率點（K 個 RB 的中心頻率），後面用 cir_to_ofdm_channel 得到 [K] 頻域通道，直接對應排程維度
            self.freqs = subcarrier_frequencies(self.K, self.W)


        # 簡化通道 baseline
        else:
            self.rayleigh = RayleighBlockFading(
                num_rx=self.A,
                num_rx_ant=self.Nr,
                num_tx=self.U,
                num_tx_ant=self.Nt
            )

        self.reset_pf_state()

        print(
            f"✅ Env Init | 3GPP={self.enable_3gpp} | "
            f"UMi | fc={self.fc/1e9:.2f}GHz | area={self.area:.0f}m | deploy={self.deployment} | "
            f"N0={self.N0:.3e} W/Hz | A={self.A}, U={self.U}, K={self.K}"
        )

    # ----------------------------
    # PF state，PF 的 denominator 初始值，避免一開始 avg_rate=0 導致 PF score 無限大；設 1.0 是常見做法
    # ----------------------------
    def reset_pf_state(self):
        self.avg_rate = tf.ones([self.U], dtype=tf.float32) * 1.0

    # ----------------------------
    # Position generators
    # ----------------------------
    def _uniform_positions(self, n: int) -> tf.Tensor:   
        return tf.random.uniform([n, 2], 0.0, self.area, dtype=tf.float32)      # 在正方形內均勻撒點

    def _street_positions(self, n: int, area: float, block: float) -> tf.Tensor:
        """
        Simple street-canyon abstraction:
        - Generate random (x,y) then snap either x or y to nearest street line (multiples of block)
        - Produces points that lie along Manhattan-like streets.
        """
        # 先亂撒點 xy
        xy = tf.random.uniform([n, 2], 0.0, area, dtype=tf.float32)   
        # snapped 把點吸到最近的街道線（block 的倍數）
        snapped = tf.round(xy / block) * block                           
        
        # 每個點隨機決定「吸 x」或「吸 y」→ 就會落在水平或垂直街道上
        snap_axis = tf.random.uniform([n], 0, 2, dtype=tf.int32)
        x = tf.where(tf.equal(snap_axis, 0), snapped[:, 0], xy[:, 0])
        y = tf.where(tf.equal(snap_axis, 1), snapped[:, 1], xy[:, 1])

        # clip 確保不超出邊界
        out = tf.stack([x, y], axis=1)
        out = tf.clip_by_value(out, 0.0, area)
        return out

    # 生成 UE 位置 + 高度 + indoor mask，同時支援 pure outdoor（indoor_prob=0）
    def sample_ue_positions(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Returns:
        - pos_2d: [U,2]
        - pos_3d: [U,3]
        - indoor_mask: [U] bool
        """
        if self.deployment == "street":
            pos_2d = self._street_positions(self.U, self.area, self.street_block_m)
        else:
            pos_2d = self._uniform_positions(self.U)

        pos_3d = tf.concat([pos_2d, tf.fill([self.U, 1], self.ue_height_m)], axis=1)

        if self.indoor_prob <= 0.0:
            indoor = tf.zeros([self.U], dtype=tf.bool)
        else:
            indoor = tf.random.uniform([self.U], 0.0, 1.0) < self.indoor_prob

        return pos_2d, pos_3d, indoor

    # ----------------------------
    # Hungarian helper (FULL NAN SAFE)，穩定性關鍵
    # ----------------------------
    def schedule_optimal_one_rb(self, score_buk: np.ndarray, L: int) -> np.ndarray:
        B, U, K = score_buk.shape
        x_out = np.zeros((B, U, K), dtype=np.float32)

        for b in range(B):
            scores = score_buk[b].astype(np.float64)

            # sanitize scores to avoid NaN/Inf crashing Hungarian
            scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

            # 把每個 RB 複製 L 次（代表每 RB 允許 L 個 UE)，把「每個 RB 可多人共享」轉成標準 assignment 問題
            cost_matrix = -np.repeat(scores, L, axis=1)
            cost_matrix = np.nan_to_num(cost_matrix, nan=1e9, posinf=1e9, neginf=1e9)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            original_k = col_ind // L
            x_out[b, row_ind, original_k] = 1.0

        return x_out

    # ----------------------------
    # Robust Topology Setter (Sionna 0.19.x safe)，相容性段
    # 準備 orientation、velocity、indoor 狀態
    # 用 try/except 去適配 Sionna 版本差異
    # 保證 set_topology 能成功
    # ----------------------------
    def _set_topology_3gpp(self, ut_pos: tf.Tensor, bs_pos: tf.Tensor, indoor: tf.Tensor):
        B = tf.shape(ut_pos)[0]

        ut_orient = tf.zeros([B, self.U, 3], dtype=tf.float32)
        bs_orient = tf.zeros([B, self.A, 3], dtype=tf.float32)

        # velocity (optional, safe)
        if self.ue_speed_mps <= 0.0:
            ut_vel = tf.zeros([B, self.U, 3], dtype=tf.float32)
        else: # 給每個 UE 一個隨機方向的速度向量
            theta = tf.random.uniform([B, self.U, 1], 0.0, 2.0 * np.pi, dtype=tf.float32)
            vx = self.ue_speed_mps * tf.cos(theta)
            vy = self.ue_speed_mps * tf.sin(theta)
            vz = tf.zeros_like(vx)
            ut_vel = tf.concat([vx, vy, vz], axis=2)

        in_state = tf.tile(tf.expand_dims(tf.cast(indoor, tf.bool), 0), [B, 1])

        # 多個 signature 嘗試 (Sionna versions differ)，避免因為 minor version API 不同就整個壞掉
        try:
            self.channel_model_3gpp.set_topology(
                ut_pos, bs_pos,
                ut_orientations=ut_orient,
                bs_orientations=bs_orient,
                ut_velocities=ut_vel,
                in_state=in_state
            )
            return
        except TypeError:
            pass

        try:
            self.channel_model_3gpp.set_topology(
                ut_pos, bs_pos,
                ut_orientations=ut_orient,
                bs_orientations=bs_orient,
                in_state=in_state
            )
            return
        except TypeError:
            pass

        try:
            self.channel_model_3gpp.set_topology(
                ut_pos, bs_pos,
                ut_orientations=ut_orient,
                bs_orientations=bs_orient,
                indoor=in_state
            )
            return
        except TypeError:
            pass

        try:
            self.channel_model_3gpp.set_topology(ut_pos, bs_pos, ut_orient, bs_orient, ut_vel, in_state)
            return
        except Exception as e:
            raise RuntimeError(f"[set_topology FAILED] Unsupported signature: {e}")

    # ----------------------------
    # Robust reshape to [B,A,Nr,U,K]
    # ----------------------------
    def _reshape_hfreq_to_BANRUK(self, h_freq: tf.Tensor) -> tf.Tensor:
        rank = tf.rank(h_freq)

        # rank=7 時 squeeze 掉 size=1 的維度
        if rank == 7:
            shape = tf.shape(h_freq)
            if shape[-1] == 1:
                h_freq = tf.squeeze(h_freq, axis=-1)
            elif shape[-2] == 1:
                h_freq = tf.squeeze(h_freq, axis=-2)

        # rank=7 時 squeeze 掉 size=1 的維度
        rank2 = tf.rank(h_freq)
        if rank2 == 6:
            # [B,A,Nr,U,Nt,K]
            if self.Nt == 1:
                h_freq = tf.squeeze(h_freq, axis=4)
            else:
                h_freq = tf.reduce_mean(h_freq, axis=4)

        # rank=5 直接 pass
        elif rank2 == 5:
            pass
        # 不符合就 raise
        else:
            raise RuntimeError(f"Unexpected h_freq rank={rank2}, shape={h_freq.shape}")

        return h_freq

    # ----------------------------
    # Channel sampling
    # ----------------------------
    def sample_channel(self, batch_size: int = 1):

        # 算每個 AP 到每個 UE 的距離，後面 cluster 需要距離，3GPP model 也能算，保留 d 可以做分析、可視化、baseline
        ue_pos_2d, ue_pos_3d, indoor = self.sample_ue_positions()
        ap_xy = tf.expand_dims(self.ap_pos_2d, axis=1)
        ue_xy = tf.expand_dims(ue_pos_2d, axis=0)
        d = tf.norm(ap_xy - ue_xy, axis=-1)  # [A,U]

        # 把 [U,3] 擴成 [B,U,3]；AP 也一樣
        if self.enable_3gpp:
            ut_pos = tf.tile(tf.expand_dims(ue_pos_3d, 0), [batch_size, 1, 1])
            bs_pos = tf.tile(tf.expand_dims(self.ap_pos_3d, 0), [batch_size, 1, 1])

            # 讓 3GPP 模型知道「UE/BS 的位置、狀態」，然後生成 CIR（a,tau）
            self._set_topology_3gpp(ut_pos, bs_pos, indoor)

            # CIR
            a, tau = self.channel_model_3gpp(num_time_samples=1, sampling_frequency=1.0)

            # CIR → OFDM 頻域通道，然後統一 shape，轉 complex64
            h_freq = cir_to_ofdm_channel(self.freqs, a, tau, normalize=False)
            h_freq = self._reshape_hfreq_to_BANRUK(h_freq)
            h = tf.cast(h_freq, tf.complex64)  # [B,A,Nr,U,K]

            # HARD NaN/Inf cleanup
            h_real = tf.math.real(h)
            h_imag = tf.math.imag(h)
            # 把任何不正常值直接歸零
            finite_mask = tf.math.is_finite(h_real) & tf.math.is_finite(h_imag)
            h = tf.where(
                finite_mask,
                h,
                tf.complex(tf.zeros_like(h_real), tf.zeros_like(h_imag))
            )

            # 用小尺度通道能量估計大尺度 β（平均 over 天線與頻率）
            beta_est = tf.reduce_mean(tf.abs(h) ** 2, axis=[2, 4])  # [B,A,U]
            beta_est = tf.where(tf.math.is_finite(beta_est), beta_est, tf.zeros_like(beta_est))

        else:
            # fallback baseline (sanity check)
            c = 299_792_458.0
            pl0 = 20.0 * tf.math.log(4.0 * np.pi * self.fc / c) / tf.math.log(10.0)

            shadow = tf.random.normal([self.A, self.U], 0.0, 8.0, dtype=tf.float32)
            pl_db = pl0 + 10.0 * 3.2 * (tf.math.log(tf.maximum(d, 10.0)) / tf.math.log(10.0)) + shadow
            beta = tf.pow(10.0, -pl_db / 10.0)

            h_raw = self.rayleigh(batch_size, num_time_steps=self.K)[0]
            h_s_flat = tf.reshape(h_raw, [batch_size, -1, self.K])

            beta_expanded = tf.expand_dims(beta, 1)
            beta_tiled = tf.tile(beta_expanded, [1, self.Nr, 1])
            beta_flat = tf.reshape(beta_tiled, [1, -1, 1])

            h_flat = tf.cast(tf.sqrt(beta_flat), tf.complex64) * tf.cast(h_s_flat, tf.complex64)
            h = tf.reshape(h_flat, [batch_size, self.A, self.Nr, self.U, self.K])

            beta_est = tf.expand_dims(beta, axis=0)
            beta_est = tf.where(tf.math.is_finite(beta_est), beta_est, tf.zeros_like(beta_est))

        g = tf.reduce_mean(tf.abs(h) ** 2, axis=2)  # [B,A,U,K]
        g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        return h, g, d, beta_est

    # ----------------------------
    # Power control (ULTRA safe)
    # 每個 UE 看自己在 serving set 中最大的 β（最強 link)，依照 fractional PC：p ~ beta^{-alpha}
    # alpha=0.6 表示「補償一部分路損」，不是全補償（避免弱 UE 直接爆功率干擾別人）
    # 提升公平性、避免超強 UE 霸佔資源
    # ----------------------------
    def fractional_power_control(self, beta_est: tf.Tensor, alpha: float = 0.6, p0_dbm: float = -10.0) -> tf.Tensor:
        beta_est = tf.where(tf.math.is_finite(beta_est), beta_est, tf.zeros_like(beta_est))

        beta_max = tf.reduce_max(beta_est, axis=1)  # [B,U]
        beta_max = tf.maximum(beta_max, 1e-12)
        beta_max = tf.where(tf.math.is_finite(beta_max), beta_max, tf.ones_like(beta_max) * 1e-12)

        p0 = tf.pow(10.0, p0_dbm / 10.0) * 1e-3
        target_p = p0 * tf.pow(beta_max, -alpha)
        target_p = tf.where(tf.math.is_finite(target_p), target_p, tf.ones_like(target_p) * self.pmax)

        final_p = tf.minimum(target_p, self.pmax)
        final_p = tf.where(tf.math.is_finite(final_p), final_p, tf.ones_like(final_p) * self.pmax)

        return tf.expand_dims(final_p, axis=-1)

    # ----------------------------
    # Cluster nearest （user-centric cell-free 的核心）
    # ----------------------------
    def get_clusters_nearest(self, d_ap_ue: tf.Tensor, C: int) -> tf.Tensor:
        _, idx = tf.math.top_k(-tf.transpose(d_ap_ue), k=C)   # 對每個 UE 選距離最近的 C 個 AP
        return tf.cast(idx, tf.int32)

    # ----------------------------
    # ✅ Sum rate (NaN safe)  ✅ FIXED einsum dims  *MRC combining 的 uplink SINR」的近似版本*
    # ----------------------------
    def compute_sum_rate_mrc(self, h, x_uk, ap_idx_uc, p_uk):
        x = tf.cast(x_uk, tf.float32)
        p_act = tf.cast(p_uk, tf.float32) * x

        B_dim = tf.shape(h)[0]
        U_dim = self.U
        K_dim = self.K

        # 把 serving set 映射到「天線維度 R = A*Nr」
        mask_ap = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=self.A, dtype=tf.float32), axis=1)
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)

        # 把通道 reshape 成 [B,U,R,K]
        h_flat = tf.reshape(h, [B_dim, self.A * self.Nr, U_dim, K_dim])
        H = tf.transpose(h_flat, [0, 2, 1, 3])  # [B,U,R,K]

        # s = sum_r |h|^2：MRC 的有效增益
        s = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)
        s = tf.cast(s, tf.float32)
        s = tf.where(tf.math.is_finite(s), s, tf.zeros_like(s))

        mask_r_c = tf.cast(mask_r, H.dtype)

        # ip：不同 UE 的內積（造成干擾）
        ip = tf.einsum('ur,burk,bvrk->buvk', mask_r_c, tf.math.conj(H), H)
        corr_pow = tf.abs(ip) ** 2
        corr_pow = tf.where(tf.math.is_finite(corr_pow), corr_pow, tf.zeros_like(corr_pow))

        total = tf.reduce_sum(tf.expand_dims(p_act, axis=1) * tf.cast(corr_pow, tf.float32), axis=2)

        # 組合 signal / interf / noise 得 SINR
        signal = p_act * (s ** 2)
        interf = total - signal
        noise = float(self.N0 * self.W) * s

        sinr = signal / (interf + noise + 1e-15)
        sinr = tf.where(tf.math.is_finite(sinr), sinr, tf.zeros_like(sinr))

        # rate = W log2(1+sinr)
        rate = self.W * (tf.math.log(1.0 + sinr) / tf.math.log(2.0))
        rate = tf.where(tf.math.is_finite(rate), rate, tf.zeros_like(rate))

        sum_rate = tf.reduce_sum(rate)
        sum_rate = tf.where(tf.math.is_finite(sum_rate), sum_rate, tf.constant(0.0, dtype=sum_rate.dtype))

        return sum_rate, rate

    # ----------------------------
    # STEP (一次 time slot)
    # 1.sample_channel → 得 h, d, beta_est
    # 2.cluster → 得每 UE 的 serving AP set ap_idx
    # 3.power control
    # 4.計算 s_gain 做 scheduling score
    # 5.依 scheduler 產生 x（UE-RB assignment）
    # 6.compute_sum_rate_mrc 評估 rate
    # 7.若 PF → 更新 avg_rate（EMA）
    # ----------------------------
    def step(self, batch_size: int, C: int, L: int, scheduler: str = "pf", enable_power_control: bool = False) -> Dict[str, Any]:
        h, g, d, beta_est = self.sample_channel(batch_size)
        ap_idx = self.get_clusters_nearest(d, C=C)

        if enable_power_control:
            p_uk = self.fractional_power_control(beta_est, alpha=0.6)
            p_uk = tf.tile(p_uk, [1, 1, self.K])
        else:
            p_uk = tf.ones([batch_size, self.U, self.K], dtype=tf.float32) * self.pmax

        B_dim = tf.shape(h)[0]

        h_flat = tf.reshape(h, [B_dim, self.A * self.Nr, self.U, self.K])
        H = tf.transpose(h_flat, [0, 2, 1, 3])  # [B,U,R,K]

        mask_ap = tf.reduce_max(tf.one_hot(ap_idx, depth=self.A, dtype=tf.float32), axis=1)
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)

        s_gain = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)
        s_gain = tf.where(tf.math.is_finite(s_gain), s_gain, tf.zeros_like(s_gain))

        # Scheduling
        if scheduler == "random":
            rng = np.random.default_rng()
            x_np = np.zeros((batch_size, self.U, self.K), dtype=np.float32)
            for b in range(batch_size):
                users = rng.permutation(self.U)[:self.K * L]
                for i, u in enumerate(users):
                    x_np[b, u, i // L] = 1.0
            x = tf.convert_to_tensor(x_np, dtype=tf.float32)

        elif scheduler == "optimal":
            x_np = self.schedule_optimal_one_rb(s_gain.numpy(), L=L)
            x = tf.convert_to_tensor(x_np, dtype=tf.float32)

        elif scheduler == "pf":
            noise = float(self.N0 * self.W)
            sinr0 = (p_uk * s_gain) / (noise + 1e-12)
            r0 = self.W * tf.math.log(1.0 + sinr0) / tf.math.log(2.0)
            r0 = tf.where(tf.math.is_finite(r0), r0, tf.zeros_like(r0))

            avg = tf.reshape(self.avg_rate + self.pf_eps, [1, self.U, 1])
            score = r0 / avg
            score = tf.where(tf.math.is_finite(score), score, tf.zeros_like(score))

            x_np = self.schedule_optimal_one_rb(score.numpy(), L=L)
            x = tf.convert_to_tensor(x_np, dtype=tf.float32)

        else:  # gain
            x_np = self.schedule_optimal_one_rb(s_gain.numpy(), L=L)
            x = tf.convert_to_tensor(x_np, dtype=tf.float32)

        sum_rate, rate = self.compute_sum_rate_mrc(h, x, ap_idx, p_uk)

        if scheduler == "pf":
            r_u = tf.reduce_mean(tf.reduce_sum(rate, axis=2), axis=0)
            r_u = tf.where(tf.math.is_finite(r_u), r_u, tf.zeros_like(r_u))
            self.avg_rate = (1.0 - self.pf_ema_alpha) * self.avg_rate + self.pf_ema_alpha * r_u

        return {"sum_rate": sum_rate, "rate": rate, "x": x}
