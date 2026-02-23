import inspect
import numpy as np
import tensorflow as tf

from sionna.channel.tr38901 import Antenna, UMi, AntennaArray
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
from utils import SimConfig, cnormal

class CellFreeNetwork:
    
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # ----------------------------
        # OFDM 參數設定
        # ----------------------------
        self.subcarrier_spacing = float(getattr(cfg, "subcarrier_spacing", 30e3)) 
        self.fft_size = int(getattr(cfg, "fft_size", 72)) 
        
        # AP positions (with Jitter)
        self.ap_pos = self._make_ap_positions_jittered()

        # Mobility params
        self.ue_speed_mps = float(getattr(cfg, "ue_speed_mps", 1.0))
        self.heading_sigma_rad = float(getattr(cfg, "heading_sigma_rad", np.pi / 8))
        self.slot_duration_s = float(getattr(cfg, "slot_duration_s", 1.0))
        self.dt_slots = 1 

        # Indoor ratio
        self.p_indoor = 0.2

        # UE: single antenna (tx)
        ut_ant = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=cfg.carrier_frequency,
        )

        # AP(BS): M-antenna array (rx), ULA: 1 x M
        bs_array = AntennaArray(
            num_rows=1,
            num_cols=int(cfg.M),
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=cfg.carrier_frequency,
        )

        # 建立 UMi Channel Model
        self.umi = UMi(
            carrier_frequency=cfg.carrier_frequency,
            o2i_model=cfg.o2i_model,
            ut_array=ut_ant,
            bs_array=bs_array,
            direction="uplink",
            enable_shadow_fading=True, 
        )

        # Persistent UE state across drops
        self._ue_pos = None
        self._ue_vel = None
        self._ue_heading = None
        self._in_state = None 

        # Cache supported kwargs for safety
        self._topo_keys = set(inspect.signature(self.umi.set_topology).parameters.keys())
        self._call_keys = set(inspect.signature(self.umi.__call__).parameters.keys())

    # ----------------------------
    # Topology: AP Jitter & User Hotspots
    # ----------------------------
    def _make_ap_positions_jittered(self) -> np.ndarray:
        cfg = self.cfg
        g = int(np.ceil(np.sqrt(cfg.n_ap)))
        xs = np.linspace(0.1 * cfg.area_side, 0.9 * cfg.area_side, g)
        ys = np.linspace(0.1 * cfg.area_side, 0.9 * cfg.area_side, g)
        pts = []
        
        step = (0.8 * cfg.area_side) / (g - 1) if g > 1 else 0
        jitter = 0.3 * step 

        for y in ys:
            for x in xs:
                if len(pts) < cfg.n_ap:
                    jx = self.rng.uniform(-jitter, jitter)
                    jy = self.rng.uniform(-jitter, jitter)
                    
                    px = np.clip(x + jx, 0, cfg.area_side)
                    py = np.clip(y + jy, 0, cfg.area_side)
                    
                    pts.append([px, py, cfg.bs_height])
        return np.array(pts, dtype=np.float32)

    def sample_ue_positions(self) -> np.ndarray:
        cfg = self.cfg
        U = int(cfg.n_ue)
        
        is_clustered = self.rng.random() < 0.5 

        if not is_clustered:
            x = self.rng.uniform(0, cfg.area_side, size=(U,))
            y = self.rng.uniform(0, cfg.area_side, size=(U,))
        else:
            n_clusters = self.rng.integers(2, 5) 
            centers = self.rng.uniform(0.1*cfg.area_side, 0.9*cfg.area_side, size=(n_clusters, 2))
            cluster_ids = self.rng.integers(0, n_clusters, size=(U,))
            
            std_dev = 25.0
            offsets = self.rng.normal(0, std_dev, size=(U, 2))
            pos = centers[cluster_ids] + offsets
            pos = np.clip(pos, 0, cfg.area_side)
            x, y = pos[:, 0], pos[:, 1]

        z = np.full((U,), cfg.ut_height, dtype=np.float32)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    # ----------------------------
    # Mobility Logic
    # ----------------------------
    def _init_mobility_state_if_needed(self):
        cfg = self.cfg
        U = int(cfg.n_ue)
        if self._ue_pos is not None:
            return

        self._ue_pos = self.sample_ue_positions()
        self._ue_heading = self.rng.uniform(0.0, 2.0 * np.pi, size=(U,)).astype(np.float32)

        vx = self.ue_speed_mps * np.cos(self._ue_heading)
        vy = self.ue_speed_mps * np.sin(self._ue_heading)
        self._ue_vel = np.stack([vx, vy, np.zeros((U,), dtype=np.float32)], axis=1).astype(np.float32)

        self._in_state = (self.rng.uniform(size=(U,)) < self.p_indoor)

    def _reflect_positions_and_velocities(self, pos_xy: np.ndarray, vel_xy: np.ndarray, L: float):
        for dim in range(2):
            mask = pos_xy[:, dim] < 0.0
            if np.any(mask):
                pos_xy[mask, dim] = -pos_xy[mask, dim]
                vel_xy[mask, dim] = -vel_xy[mask, dim]

            mask = pos_xy[:, dim] > L
            if np.any(mask):
                pos_xy[mask, dim] = 2.0 * L - pos_xy[mask, dim]
                vel_xy[mask, dim] = -vel_xy[mask, dim]
                
        return pos_xy, vel_xy

    def _step_mobility_one_slot(self):
        cfg = self.cfg
        U = int(cfg.n_ue)
        L = float(cfg.area_side)

        dtheta = self.rng.normal(0.0, self.heading_sigma_rad, size=(U,)).astype(np.float32)
        self._ue_heading = (self._ue_heading + dtheta).astype(np.float32)

        self._ue_vel[:, 0] = self.ue_speed_mps * np.cos(self._ue_heading)
        self._ue_vel[:, 1] = self.ue_speed_mps * np.sin(self._ue_heading)
        self._ue_vel[:, 2] = 0.0

        dt = float(self.slot_duration_s * self.dt_slots)
        pos_xy = self._ue_pos[:, 0:2].astype(np.float32)
        vel_xy = self._ue_vel[:, 0:2].astype(np.float32)
        pos_xy = pos_xy + vel_xy * dt

        pos_xy, vel_xy = self._reflect_positions_and_velocities(pos_xy, vel_xy, L)
        self._ue_pos[:, 0:2] = pos_xy
        self._ue_vel[:, 0:2] = vel_xy
        self._ue_pos[:, 2] = float(cfg.ut_height)

    @staticmethod
    def _zeros_3vec(batch_size: int, n: int) -> tf.Tensor:
        return tf.zeros([batch_size, n, 3], dtype=tf.float32)

    def _set_topology_safe(self, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in self._topo_keys}
        return self.umi.set_topology(**filtered)

    def _call_umi_safe(self, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in self._call_keys}
        return self.umi(**filtered)

    # ----------------------------
    # TF Function for Speed
    # ----------------------------
    @tf.function(jit_compile=True)
    def _compute_channel_tf(self, frequencies, a, tau):
        """
        利用 TF Graph Mode 加速 CIR 轉 OFDM 頻域通道的過程
        """
        # cir_to_ofdm_channel 回傳: [Batch, Num_Rx, Num_Rx_Ant, Num_Tx, Num_Tx_Ant, Num_Subcarriers, Num_Time_Steps]
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
        return h_freq

    # ----------------------------
    # CIR utilities & Beta Calculation
    # ----------------------------
    @staticmethod
    def _beta_from_cir(a: tf.Tensor) -> tf.Tensor:
        a = tf.cast(a, tf.complex64)
        power = tf.abs(a) ** 2  # [B, A, M, U, 1, P, T]
        power_sum_paths = tf.reduce_sum(power, axis=5) # Sum over paths
        
        # Mean over Time(5), RxAnt(2), TxAnt(4), Batch(0) -> [A, U]
        beta = tf.reduce_mean(power_sum_paths, axis=[0, 2, 4, 5])
        
        beta = tf.cast(beta, tf.float32)
        beta = tf.maximum(beta, 1e-12)
        return beta

    @staticmethod
    def _extract_a_tau_from_umi_return(ret):
        if hasattr(ret, "a") and hasattr(ret, "tau"):
            return ret.a, ret.tau
        
        if isinstance(ret, (tuple, list)):
            tensors = [x for x in ret if isinstance(x, tf.Tensor)]
            a = next((t for t in tensors if t.dtype.is_complex), None)
            if a is None:
                raise RuntimeError("UMi tuple return missing complex tensor 'a'")
            
            tau = next((t for t in tensors if (t.dtype.is_floating or t.dtype.is_integer) and t.shape.rank == a.shape.rank), None)
            if tau is None: 
                 tau = next((t for t in tensors if (t.dtype.is_floating or t.dtype.is_integer)), None)
            
            if tau is None:
                raise RuntimeError("UMi tuple return missing 'tau'")
            return a, tau
        raise RuntimeError(f"Unknown UMi return type: {type(ret)}")

    @staticmethod
    def _extract_H_AUM_Freq(h_freq_tf: tf.Tensor, A: int, U: int, M: int, F: int) -> np.ndarray:
        """
        [維度修正版]
        Sionna output shape: [Batch, Rx(A), RxAnt(M), Tx(U), TxAnt(1), Subcarriers(F), Time(1)]
        indices:              0       1       2         3      4          5               6
        Target: [A, U, M, F]
        """
        h = h_freq_tf.numpy()
        
        # 暴力切片，確保只取我們需要的維度 (假設 Batch=0, TxAnt=0, Time=0 都是單一值)
        # 取: [0, :, :, :, 0, :, 0] -> 剩下 [A, M, U, F]
        # 注意: 這裡假設 Batch=1, TxAnt=1, Time=1，這是本模擬的預設值
        h_squeezed = h[0, :, :, :, 0, :, 0] # Shape: [A, M, U, F]
        
        # 檢查是否真的變成 4 維
        if h_squeezed.ndim != 4:
            # 防呆 fallback: 若維度不對，嘗試自動 squeeze
            h_squeezed = np.squeeze(h)
            
            # 如果 squeeze 後維度還不對 (例如 F=1 時 squeeze 會把 F 吃掉)，則報錯
            if h_squeezed.ndim != 4:
                # 嘗試 reshape 強制對齊 [A, M, U, F]
                try:
                    h_squeezed = h_squeezed.reshape(A, M, U, F)
                except ValueError:
                    raise ValueError(f"Channel H shape mismatch! Expected [A,M,U,F] but got {h_squeezed.shape} from raw {h.shape}")

        # Current shape: [A, M, U, F]
        # Target shape: [A, U, M, F] -> Swap axis 1 (M) and 2 (U)
        h_final = np.moveaxis(h_squeezed, 1, 2)
        
        return h_final.astype(np.complex64)

    # ----------------------------
    # Channel generation (Batch)
    # ----------------------------
    def generate_channel_and_beta_sionna(self, ue_pos: np.ndarray, ue_vel: np.ndarray, in_state_np):
        cfg = self.cfg
        A, U, M = int(cfg.n_ap), int(cfg.n_ue), int(cfg.M)
        B = 1

        # Prepare Tensors
        ut_loc = tf.constant(ue_pos[None, :, :], dtype=tf.float32)          # [1,U,3]
        bs_loc = tf.constant(self.ap_pos[None, :, :], dtype=tf.float32)     # [1,A,3]
        in_state = tf.constant(in_state_np[None, :], dtype=tf.bool)         # [1,U]
        
        ut_orientations = self._zeros_3vec(B, U)
        bs_orientations = self._zeros_3vec(B, A)
        ut_velocities = tf.constant(ue_vel[None, :, :], dtype=tf.float32)
        bs_velocities = self._zeros_3vec(B, A)

        self._set_topology_safe(
            ut_loc=ut_loc, bs_loc=bs_loc, in_state=in_state,
            ut_orientations=ut_orientations, bs_orientations=bs_orientations,
            ut_velocities=ut_velocities, bs_velocities=bs_velocities,
        )

        num_time_samples = int(getattr(cfg, "num_time_samples", 1))
        sampling_frequency = float(getattr(cfg, "sampling_frequency", 30.72e6))

        ret = self._call_umi_safe(
            num_time_samples=num_time_samples,
            sampling_frequency=sampling_frequency,
            return_amps=True,
            return_delays=True,
            return_path_gains=True,
        )

        a, tau = self._extract_a_tau_from_umi_return(ret)

        # 1. 計算 Beta (Large Scale Fading) - 與頻率無關
        beta = self._beta_from_cir(a).numpy().astype(np.float32)

        # 2. 計算 H (Frequency Selective) - [NEW]
        frequencies = subcarrier_frequencies(
            num_subcarriers=self.fft_size, 
            subcarrier_spacing=self.subcarrier_spacing
        )
        
        h_freq_tf = self._compute_channel_tf(frequencies, a, tau)
        
        # 整理維度 -> [A, U, M, F] (這裡會呼叫修正後的 extract)
        H = self._extract_H_AUM_Freq(h_freq_tf, A=A, U=U, M=M, F=self.fft_size)

        return beta, H

    # ----------------------------
    # Pilot assignment
    # ----------------------------
    def assign_pilots_greedy(self, beta: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        U = int(cfg.n_ue)
        tau_p = int(cfg.tau_p)

        pilot_id = np.full((U,), -1, dtype=np.int32)
        contamination_matrix = beta.T @ beta 

        ue_power = np.sum(beta, axis=0)
        sort_idx = np.argsort(-ue_power)

        pilot_groups = [[] for _ in range(tau_p)]
        for u in sort_idx:
            best_p = 0
            min_cost = 1e30
            for p in range(tau_p):
                cost = 0.0
                for v in pilot_groups[p]:
                    cost += contamination_matrix[int(u), int(v)]
                if cost < min_cost:
                    min_cost = cost
                    best_p = p
            pilot_id[int(u)] = int(best_p)
            pilot_groups[int(best_p)].append(int(u))

        pilot_id[pilot_id < 0] = 0
        pilot_id[pilot_id >= tau_p] = tau_p - 1
        return pilot_id

    # ----------------------------
    # Pilot MMSE estimate
    # ----------------------------
    def pilot_mmse_estimate(self, H: np.ndarray, beta: np.ndarray, pilot_id: np.ndarray):
        """
        支援 [A, U, M, F] 維度的 MMSE 估測
        """
        cfg = self.cfg
        
        # [安全解包]
        # 這裡現在應該保證是 4 個值了。如果還是報錯，H.shape 本身有問題。
        try:
            A, U, M, F = H.shape
        except ValueError:
             raise ValueError(f"H shape invalid inside pilot_mmse_estimate: {H.shape}. Expected 4 dimensions [A, U, M, F].")
        
        pilot_id = np.asarray(pilot_id, dtype=np.int64).copy()
        pilot_id[pilot_id < 0] = 0

        sigma2 = float(cfg.noise_power_lin)
        tau_p_sys = float(cfg.tau_p) 
        tau_p_cur = int(max(int(cfg.tau_p), 1))

        denom = np.full((A, tau_p_cur), sigma2, dtype=np.float32)
        
        Y = np.zeros((A, tau_p_cur, M, F), dtype=np.complex64)
        noise = cnormal((A, tau_p_cur, M, F), self.rng) * np.sqrt(sigma2).astype(np.float32)

        used_pilots = np.unique(pilot_id)

        for t in used_pilots:
            t = int(t)
            if t < 0: continue
            
            if t >= tau_p_cur:
                new_tau = t + 1
                denom_new = np.full((A, new_tau), sigma2, dtype=np.float32)
                denom_new[:, :tau_p_cur] = denom
                denom = denom_new

                Y_new = np.zeros((A, new_tau, M, F), dtype=np.complex64)
                Y_new[:, :tau_p_cur, :, :] = Y
                Y = Y_new

                noise_new = cnormal((A, new_tau, M, F), self.rng) * np.sqrt(sigma2).astype(np.float32)
                noise_new[:, :tau_p_cur, :, :] += noise
                noise = noise_new

                tau_p_cur = new_tau

            users_t = np.where(pilot_id == t)[0]
            if users_t.size == 0:
                continue

            sp = float(np.sqrt(tau_p_sys * cfg.p_pilot))
            
            # H[:, users_t, :, :] shape is [A, Nu, M, F] -> sum over Nu -> [A, M, F]
            Y[:, t, :, :] = sp * np.sum(H[:, users_t, :, :], axis=1)
            
            # Denom: sum(beta) -> [A]
            denom[:, t] = (tau_p_sys * cfg.p_pilot * np.sum(beta[:, users_t], axis=1) + sigma2).astype(np.float32)

        Y = Y + noise

        # Estimate Hhat
        sp = float(np.sqrt(tau_p_sys * cfg.p_pilot))
        Hhat = np.zeros_like(H, dtype=np.complex64) # [A, U, M, F]

        for u in range(U):
            t = int(pilot_id[u])
            if t < 0: t = 0
            if t >= tau_p_cur: t = tau_p_cur - 1

            # c shape: [A] -> expand to [A, 1, 1] for broadcasting
            c = (sp * beta[:, u] / (denom[:, t] + 1e-12)).astype(np.float32)
            
            # Hhat calculation
            Hhat[:, u, :, :] = (c[:, None, None].astype(np.complex64) * Y[:, t, :, :])

        return Hhat

    # ----------------------------
    # Public API
    # ----------------------------
    def generate_drop(self, pilot_method="greedy"):
        self._init_mobility_state_if_needed()
        self._step_mobility_one_slot()

        ue_pos = self._ue_pos.copy()
        ue_vel = self._ue_vel.copy()
        in_state_np = self._in_state.copy()

        beta, H = self.generate_channel_and_beta_sionna(ue_pos, ue_vel, in_state_np)

        if pilot_method == "greedy":
            pilot_id = self.assign_pilots_greedy(beta)
        else:
            pilot_id = self.rng.integers(
                low=0, high=int(self.cfg.tau_p),
                size=(int(self.cfg.n_ue),), dtype=np.int32
            )

        Hhat = self.pilot_mmse_estimate(H, beta, pilot_id)
        return ue_pos, self.ap_pos.copy(), beta, pilot_id, H, Hhat