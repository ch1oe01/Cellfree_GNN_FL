# wirelessNetwork.py
import numpy as np
import tensorflow as tf
from sionna.channel.tr38901 import Antenna, UMi
from utils import SimConfig, cnormal

class CellFreeNetwork:
    """
    產生 drop：
      - AP/UE 位置
      - beta[a,u] 由 Sionna UMi CIR average power 得到
      - 小尺度通道 h[a,u,m] ~ CN(0, beta[a,u])
      - pilot reuse：隨機指派 tau_p 個 pilot
      - 以 pilot observation 做 MMSE estimate：hat_h[a,u,:] = c[a,u] * y[a,pilot(u),:]
        => pilot contamination 自然形成（共享 pilot 的估測相關）
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.ap_pos = self._make_ap_positions_grid()

        ut_ant = Antenna(polarization="single", polarization_type="V",
                         antenna_pattern="omni", carrier_frequency=cfg.carrier_frequency)
        bs_ant = Antenna(polarization="single", polarization_type="V",
                         antenna_pattern="omni", carrier_frequency=cfg.carrier_frequency)
        self.umi = UMi(carrier_frequency=cfg.carrier_frequency,
                       o2i_model=cfg.o2i_model,
                       ut_array=ut_ant, bs_array=bs_ant,
                       direction="uplink")

    def _make_ap_positions_grid(self) -> np.ndarray:
        cfg = self.cfg
        g = int(np.ceil(np.sqrt(cfg.n_ap)))
        xs = np.linspace(0.1*cfg.area_side, 0.9*cfg.area_side, g)
        ys = np.linspace(0.1*cfg.area_side, 0.9*cfg.area_side, g)
        pts = []
        for y in ys:
            for x in xs:
                if len(pts) < cfg.n_ap:
                    pts.append([x, y, cfg.bs_height])
        return np.array(pts, dtype=np.float32)

    def sample_ue_positions(self) -> np.ndarray:
        cfg = self.cfg
        x = self.rng.uniform(0, cfg.area_side, size=(cfg.n_ue,))
        y = self.rng.uniform(0, cfg.area_side, size=(cfg.n_ue,))
        z = np.full((cfg.n_ue,), cfg.ut_height)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def beta_from_sionna(self, ue_pos: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        A, U = cfg.n_ap, cfg.n_ue
        beta = np.zeros((A, U), dtype=np.float32)

        B = 1
        ut_loc = tf.constant(ue_pos[None, :, :], dtype=tf.float32)  # [1,U,3]
        ut_or = tf.zeros([B, U, 3], dtype=tf.float32)
        ut_vel = tf.zeros([B, U, 3], dtype=tf.float32)
        in_state = tf.zeros([B, U], dtype=tf.bool)

        for a in range(A):
            bs_loc = tf.constant(self.ap_pos[None, None, a, :], dtype=tf.float32)  # [1,1,3]
            bs_or = tf.zeros([B, 1, 3], dtype=tf.float32)

            self.umi.set_topology(
                ut_loc=ut_loc,
                bs_loc=bs_loc,
                ut_orientations=ut_or,
                bs_orientations=bs_or,
                ut_velocities=ut_vel,
                in_state=in_state,
                los=None
            )

            a_cir, _tau = self.umi(1, 200)
            a_np = a_cir.numpy()
            pow_ = np.mean(np.abs(a_np)**2, axis=(0,1,2,4,5,6))  # -> [U]
            beta[a, :] = np.maximum(pow_.astype(np.float32), 1e-12)

        return beta

    def assign_pilots(self) -> np.ndarray:
        """pilot index for each UE: int in [0, tau_p-1]"""
        cfg = self.cfg
        return self.rng.integers(low=0, high=cfg.tau_p, size=(cfg.n_ue,), dtype=np.int32)

    def small_scale_channel(self, beta: np.ndarray) -> np.ndarray:
        """h[a,u,m] ~ CN(0, beta[a,u])"""
        cfg = self.cfg
        g = cnormal((cfg.n_ap, cfg.n_ue, cfg.M), self.rng)
        return g * np.sqrt(beta[:, :, None]).astype(np.float32)

    def pilot_mmse_estimate(self, H: np.ndarray, beta: np.ndarray, pilot_id: np.ndarray):
        """
        pilot observation at each AP for each pilot t:
          y[a,t,:] = sum_{u:pilot(u)=t} sqrt(tau_p*p_pilot) * h[a,u,:] + n
        MMSE estimate:
          hat_h[a,u,:] = c[a,u] * y[a,t,:]
          c[a,u] = sqrt(tau_p*p_pilot)*beta[a,u] / (tau_p*p_pilot*sum_{i in U_t} beta[a,i] + sigma2)
        """
        cfg = self.cfg
        A, U, M = cfg.n_ap, cfg.n_ue, cfg.M
        tau_p = cfg.tau_p
        sp = np.sqrt(cfg.tau_p * cfg.p_pilot).astype(np.float32)
        sigma2 = cfg.noise_power_lin

        # y[a,t,m]
        Y = np.zeros((A, tau_p, M), dtype=np.complex64)
        noise = cnormal((A, tau_p, M), self.rng) * np.sqrt(sigma2).astype(np.float32)

        for t in range(tau_p):
            users_t = np.where(pilot_id == t)[0]
            if len(users_t) == 0:
                continue
            # sum of true channels
            Y[:, t, :] = sp * np.sum(H[:, users_t, :], axis=1)
        Y = Y + noise

        # denominator per (a,t): tau_p*p_pilot*sum beta + sigma2
        denom = np.zeros((A, tau_p), dtype=np.float32)
        for t in range(tau_p):
            users_t = np.where(pilot_id == t)[0]
            if len(users_t) == 0:
                denom[:, t] = sigma2
            else:
                denom[:, t] = (cfg.tau_p * cfg.p_pilot * np.sum(beta[:, users_t], axis=1) + sigma2).astype(np.float32)

        # estimate
        Hhat = np.zeros_like(H, dtype=np.complex64)
        for u in range(U):
            t = int(pilot_id[u])
            c = (sp * beta[:, u] / (denom[:, t] + 1e-12)).astype(np.float32)  # [A]
            Hhat[:, u, :] = (c[:, None].astype(np.complex64) * Y[:, t, :])

        return Hhat  # [A,U,M] complex64

    def generate_drop(self):
        """
        Returns:
          ue_pos [U,3], ap_pos [A,3], beta [A,U], pilot_id [U], H [A,U,M], Hhat [A,U,M]
        """
        ue_pos = self.sample_ue_positions()
        beta = self.beta_from_sionna(ue_pos)
        pilot_id = self.assign_pilots()
        H = self.small_scale_channel(beta)
        Hhat = self.pilot_mmse_estimate(H, beta, pilot_id)
        return ue_pos, self.ap_pos.copy(), beta, pilot_id, H, Hhat
