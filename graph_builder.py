# graph_builder.py
import tensorflow as tf

def build_bipartite_features(
    g_ap_ue_rb: tf.Tensor,     # [B,A,U,K]
    ap_idx_uc: tf.Tensor,      # [U,C]
    eps: float = 1e-30
) -> dict:
    """
    回傳一個 dict，供模型使用：
      edge_feat: [B,U,A,K]  (log gain; UE->AP edges)
      cluster_mask_ua: [U,A] (UE 的 cluster AP mask)
    """
    # edge_feat 用 log(g)
    edge_feat = tf.math.log(tf.transpose(g_ap_ue_rb, [0, 2, 1, 3]) + eps)  # [B,U,A,K]

    # cluster mask [U,A]
    cluster_mask_ua = tf.reduce_max(tf.one_hot(ap_idx_uc, depth=tf.shape(g_ap_ue_rb)[1], dtype=tf.float32), axis=1)

    return {
        "edge_feat": tf.cast(edge_feat, tf.float32),
        "cluster_mask_ua": tf.cast(cluster_mask_ua, tf.float32),
    }
