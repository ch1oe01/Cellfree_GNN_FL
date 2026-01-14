# train_imitation.py
import tensorflow as tf
from configs import EnvConfig, TrainConfig, ModelConfig
from channel_env import SionnaCellFreeUplinkEnv
from graph_builder import build_bipartite_features
from models import BipartiteSchedulerGNN, topk_mask

def bce_with_logits(labels, logits, label_smoothing=0.0):
    if label_smoothing > 0:
        labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

def main():
    env_cfg = EnvConfig()
    tr_cfg = TrainConfig()
    m_cfg = ModelConfig()

    env = SionnaCellFreeUplinkEnv(**env_cfg.__dict__)
    model = BipartiteSchedulerGNN(hidden=m_cfg.hidden, msg_hidden=m_cfg.msg_hidden, num_layers=m_cfg.num_layers)
    opt = tf.keras.optimizers.Adam(tr_cfg.lr)

    @tf.function
    def train_step():
        h, g, d = env.sample_channel(batch_size=tr_cfg.batch_size)      # h:[B,A,Nr,U,K], g:[B,A,U,K]
        ap_idx_uc = env.get_clusters_nearest(d, C=tr_cfg.C)             # [U,C]

        # baseline label
        x_label = env.schedule_topL_by_cluster_gain(h, ap_idx_uc, L=tr_cfg.L)  # [B,U,K]

        feat = build_bipartite_features(g, ap_idx_uc)
        edge_feat = feat["edge_feat"]                                   # [B,U,A,K]

        with tf.GradientTape() as tape:
            scores = model(edge_feat, training=True)                    # [B,U,K] logits-ish
            loss = bce_with_logits(x_label, scores, tr_cfg.label_smoothing)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        # evaluate learned schedule rate (hard top-L)
        x_pred = topk_mask(scores, L=tr_cfg.L)
        sum_rate, _, _ = env.compute_sum_rate_mrc_clustered(h, x_pred, ap_idx_uc)
        sum_rate_label, _, _ = env.compute_sum_rate_mrc_clustered(h, x_label, ap_idx_uc)

        return loss, sum_rate, sum_rate_label

    for step in range(1, tr_cfg.steps + 1):
        loss, sr_pred, sr_label = train_step()
        if step % tr_cfg.print_every == 0:
            tf.print(
                "step", step,
                "| loss", loss,
                "| SR(pred)", sr_pred,
                "| SR(label)", sr_label
            )

if __name__ == "__main__":
    main()
