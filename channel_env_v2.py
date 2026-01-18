# channel_env_v2.py
import tensorflow as tf
import numpy as np
import sionna

from sionna.channel import RayleighBlockFading
from sionna.channel.tr38901 import UMa, Antenna, AntennaArray
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any


class SionnaCellFreeUplinkEnvV2:
    """
    ✅ FINAL Stable Version (Sionna 0.19.2 + GTX1060 OK)
    ------------------------------------------------------------
    Fixes:
    - 3GPP UMa API compatibility
    - Robust reshape of h_freq -> [B,A,Nr,U,K]
    - Full NaN/Inf guards for PC=True
    - Hungarian scheduler won't crash (nan_to_num)
    - ✅ FIX einsum rank mismatch: use 'ur,burk->buk'
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
        rb_bw_hz: float = 180e3,
        noise_psd_w_per_hz: float = 1e-20,
        pmax_watt: float = 0.2,
        seed: int = 7,
        debug: bool = False,
        pf_ema_alpha: float = 0.02,
        pf_eps: float = 1e-6,
        enable_3gpp: bool = False,
    ):
        self.A = int(num_ap)
        self.U = int(num_ue)
        self.K = int(num_rb)

        self.area = float(area_side_m)
        self.fc = float(carrier_freq_hz)

        self.Nr = int(num_ap_rx_ant)
        self.Nt = int(num_ue_tx_ant)

        self.W = float(rb_bw_hz)
        self.N0 = float(noise_psd_w_per_hz)
        self.pmax = float(pmax_watt)

        self.debug = bool(debug)
        self.pf_ema_alpha = float(pf_ema_alpha)
        self.pf_eps = float(pf_eps)
        self.enable_3gpp = bool(enable_3gpp)

        tf.random.set_seed(int(seed))
        self._np_rng = np.random.default_rng(int(seed))

        # AP positions
        self.ap_pos_2d = self._uniform_positions(self.A)  # [A,2]
        self.ap_pos_3d = tf.concat([self.ap_pos_2d, tf.fill([self.A, 1], 25.0)], axis=1)  # [A,3]

        # ----------------------------
        # Channel Model
        # ----------------------------
        if self.enable_3gpp:
            print(f"🚀 Initializing 3GPP UMa Channel (Sionna v{sionna.__version__})...")

            # UE Array
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

            # BS Array
            rows = int(np.sqrt(self.Nr))
            rows = max(rows, 1)
            cols = int(np.ceil(self.Nr / rows))
            cols = max(cols, 1)

            bs_array = AntennaArray(
                num_rows=rows,
                num_cols=cols,
                polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=self.fc
            )

            self.channel_model_3gpp = UMa(
                carrier_frequency=self.fc,
                o2i_model="low",
                ut_array=ut_array,
                bs_array=bs_array,
                direction="uplink",
                enable_pathloss=True,
                enable_shadow_fading=True
            )

            self.freqs = subcarrier_frequencies(self.K, self.W)

        else:
            self.rayleigh = RayleighBlockFading(
                num_rx=self.A,
                num_rx_ant=self.Nr,
                num_tx=self.U,
                num_tx_ant=self.Nt
            )

        self.reset_pf_state()
        print(f"✅ Env Init | 3GPP={self.enable_3gpp} | BW/RB={self.W/1e3:.1f}kHz | A={self.A}, U={self.U}, K={self.K}")

    def reset_pf_state(self):
        self.avg_rate = tf.ones([self.U], dtype=tf.float32) * 1.0

    def _uniform_positions(self, n: int) -> tf.Tensor:
        return tf.random.uniform([n, 2], 0.0, self.area, dtype=tf.float32)

    def sample_ue_positions(self):
        pos_2d = self._uniform_positions(self.U)
        pos_3d = tf.concat([pos_2d, tf.fill([self.U, 1], 1.5)], axis=1)
        return pos_2d, pos_3d

    # ----------------------------
    # ✅ Hungarian helper (FULL NAN SAFE)
    # ----------------------------
    def schedule_optimal_one_rb(self, score_buk: np.ndarray, L: int) -> np.ndarray:
        B, U, K = score_buk.shape
        x_out = np.zeros((B, U, K), dtype=np.float32)

        for b in range(B):
            scores = score_buk[b].astype(np.float64)

            # ✅ sanitize scores to avoid NaN/Inf crashing Hungarian
            scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

            cost_matrix = -np.repeat(scores, L, axis=1)
            cost_matrix = np.nan_to_num(cost_matrix, nan=1e9, posinf=1e9, neginf=1e9)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            original_k = col_ind // L
            x_out[b, row_ind, original_k] = 1.0

        return x_out

    # ----------------------------
    # ✅ Robust Topology Setter (Sionna 0.19.x safe)
    # ----------------------------
    def _set_topology_3gpp(self, ut_pos: tf.Tensor, bs_pos: tf.Tensor):
        B = tf.shape(ut_pos)[0]

        ut_orient = tf.zeros([B, self.U, 3], dtype=tf.float32)
        bs_orient = tf.zeros([B, self.A, 3], dtype=tf.float32)
        ut_vel = tf.zeros([B, self.U, 3], dtype=tf.float32)
        in_state = tf.zeros([B, self.U], dtype=tf.bool)

        # Try various signatures (Sionna versions differ)
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
    # ✅ Robust reshape to [B,A,Nr,U,K]
    # ----------------------------
    def _reshape_hfreq_to_BANRUK(self, h_freq: tf.Tensor) -> tf.Tensor:
        rank = tf.rank(h_freq)

        if rank == 7:
            shape = tf.shape(h_freq)
            if shape[-1] == 1:
                h_freq = tf.squeeze(h_freq, axis=-1)
            elif shape[-2] == 1:
                h_freq = tf.squeeze(h_freq, axis=-2)

        rank2 = tf.rank(h_freq)

        if rank2 == 6:
            # [B,A,Nr,U,Nt,K]
            if self.Nt == 1:
                h_freq = tf.squeeze(h_freq, axis=4)
            else:
                h_freq = tf.reduce_mean(h_freq, axis=4)

        elif rank2 == 5:
            # already [B,A,Nr,U,K]
            pass
        else:
            raise RuntimeError(f"Unexpected h_freq rank={rank2}, shape={h_freq.shape}")

        return h_freq

    # ----------------------------
    # Channel sampling
    # ----------------------------
    def sample_channel(self, batch_size: int = 1):
        ue_pos_2d, ue_pos_3d = self.sample_ue_positions()

        ap_xy = tf.expand_dims(self.ap_pos_2d, axis=1)
        ue_xy = tf.expand_dims(ue_pos_2d, axis=0)
        d = tf.norm(ap_xy - ue_xy, axis=-1)  # [A,U]

        if self.enable_3gpp:
            ut_pos = tf.tile(tf.expand_dims(ue_pos_3d, 0), [batch_size, 1, 1])
            bs_pos = tf.tile(tf.expand_dims(self.ap_pos_3d, 0), [batch_size, 1, 1])

            self._set_topology_3gpp(ut_pos, bs_pos)

            # CIR
            a, tau = self.channel_model_3gpp(num_time_samples=1, sampling_frequency=1.0)

            # OFDM channel
            h_freq = cir_to_ofdm_channel(self.freqs, a, tau, normalize=False)
            h_freq = self._reshape_hfreq_to_BANRUK(h_freq)
            h = tf.cast(h_freq, tf.complex64)  # [B,A,Nr,U,K]

            # ✅ HARD NaN/Inf cleanup
            h_real = tf.math.real(h)
            h_imag = tf.math.imag(h)
            finite_mask = tf.math.is_finite(h_real) & tf.math.is_finite(h_imag)
            h = tf.where(finite_mask,
                         h,
                         tf.complex(tf.zeros_like(h_real), tf.zeros_like(h_imag)))

            beta_est = tf.reduce_mean(tf.abs(h)**2, axis=[2, 4])  # [B,A,U]
            beta_est = tf.where(tf.math.is_finite(beta_est), beta_est, tf.zeros_like(beta_est))

        else:
            # Legacy Rayleigh + safe flatten
            c = 299_792_458.0
            pl0 = 20.0 * tf.math.log(4.0 * np.pi * self.fc / c) / tf.math.log(10.0)

            shadow = tf.random.normal([self.A, self.U], 0.0, 8.0, dtype=tf.float32)
            pl_db = pl0 + 10.0 * 3.2 * (tf.math.log(tf.maximum(d, 5.0)) / tf.math.log(10.0)) + shadow
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
    # ✅ Power control (ULTRA safe)
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
    # Cluster nearest
    # ----------------------------
    def get_clusters_nearest(self, d_ap_ue: tf.Tensor, C: int) -> tf.Tensor:
        _, idx = tf.math.top_k(-tf.transpose(d_ap_ue), k=C)
        return tf.cast(idx, tf.int32)

    # ----------------------------
    # ✅ Sum rate (NaN safe)  ✅ FIXED einsum dims
    # ----------------------------
    def compute_sum_rate_mrc(self, h, x_uk, ap_idx_uc, p_uk):
        x = tf.cast(x_uk, tf.float32)
        p_act = tf.cast(p_uk, tf.float32) * x

        B_dim = tf.shape(h)[0]
        U_dim = self.U
        K_dim = self.K

        # mask_ap: [U,A], mask_r: [U, A*Nr]
        mask_ap = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=self.A, dtype=tf.float32), axis=1)
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)

        h_flat = tf.reshape(h, [B_dim, self.A * self.Nr, U_dim, K_dim])
        H = tf.transpose(h_flat, [0, 2, 1, 3])  # [B,U,R,K]

        # ✅ FIX: 'ur,burk->buk'  (mask_r is rank2)
        s = tf.einsum('ur,burk->buk', mask_r, tf.abs(H) ** 2)
        s = tf.cast(s, tf.float32)
        s = tf.where(tf.math.is_finite(s), s, tf.zeros_like(s))

        mask_r_c = tf.cast(mask_r, H.dtype)

        # ✅ FIX: 'ur,burk,bvrk->buvk'
        ip = tf.einsum('ur,burk,bvrk->buvk', mask_r_c, tf.math.conj(H), H)
        corr_pow = tf.abs(ip) ** 2
        corr_pow = tf.where(tf.math.is_finite(corr_pow), corr_pow, tf.zeros_like(corr_pow))

        total = tf.reduce_sum(tf.expand_dims(p_act, axis=1) * tf.cast(corr_pow, tf.float32), axis=2)

        signal = p_act * (s ** 2)
        interf = total - signal
        noise = float(self.N0 * self.W) * s

        sinr = signal / (interf + noise + 1e-15)
        sinr = tf.where(tf.math.is_finite(sinr), sinr, tf.zeros_like(sinr))

        rate = self.W * (tf.math.log(1.0 + sinr) / tf.math.log(2.0))
        rate = tf.where(tf.math.is_finite(rate), rate, tf.zeros_like(rate))

        sum_rate = tf.reduce_sum(rate)
        sum_rate = tf.where(tf.math.is_finite(sum_rate), sum_rate, tf.constant(0.0, dtype=sum_rate.dtype))

        return sum_rate, rate

    # ----------------------------
    # STEP  ✅ FIXED einsum dims
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

        # Flatten channel
        h_flat = tf.reshape(h, [B_dim, self.A * self.Nr, self.U, self.K])
        H = tf.transpose(h_flat, [0, 2, 1, 3])  # [B,U,R,K]

        # mask_ap: [U,A], mask_r: [U,R]
        mask_ap = tf.reduce_max(tf.one_hot(ap_idx, depth=self.A, dtype=tf.float32), axis=1)
        mask_r = tf.repeat(mask_ap, repeats=self.Nr, axis=1)

        # ✅ FIX: mask_r rank2
        s_gain = tf.einsum('ur,burk->buk', mask_r, tf.abs(H)**2)
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
