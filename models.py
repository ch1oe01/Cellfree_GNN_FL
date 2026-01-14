# models.py
import tensorflow as tf

class BipartiteSchedulerGNN(tf.keras.Model):
    """
    輸入：
      edge_feat: [B,U,A,K] (log g)
    輸出：
      scores: [B,U,K] (越大越應該被排上該 RB)
    """
    def __init__(self, hidden=64, msg_hidden=64, num_layers=2):
        super().__init__()
        self.hidden = int(hidden)
        self.msg_hidden = int(msg_hidden)
        self.num_layers = int(num_layers)

        # edge MLP：把每條 (u,a,k) edge 的特徵轉成 message
        self.edge_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.msg_hidden, activation="relu"),
            tf.keras.layers.Dense(self.msg_hidden, activation="relu"),
        ])

        # UE 更新 MLP
        self.ue_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden, activation="relu"),
            tf.keras.layers.Dense(self.hidden, activation="relu"),
        ])

        # 輸出 score
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, edge_feat: tf.Tensor, training=False) -> tf.Tensor:
        # edge_feat: [B,U,A,K]
        B = tf.shape(edge_feat)[0]
        U = tf.shape(edge_feat)[1]
        A = tf.shape(edge_feat)[2]
        K = tf.shape(edge_feat)[3]

        # 把 (B,U,A,K,1) 拉平做 MLP
        x = tf.expand_dims(edge_feat, axis=-1)            # [B,U,A,K,1]
        x = tf.reshape(x, [B*U*A*K, 1])                   # [B*U*A*K,1]
        m = self.edge_mlp(x, training=training)           # [B*U*A*K, msg_hidden]
        m = tf.reshape(m, [B, U, A, K, self.msg_hidden])  # [B,U,A,K,Hm]

        # 聚合：sum over AP
        agg = tf.reduce_sum(m, axis=2)                    # [B,U,K,Hm]

        # UE MLP：把每個 (u,k) 的聚合訊息轉 embedding
        ue_in = tf.reshape(agg, [B*U*K, self.msg_hidden])
        ue_h = self.ue_mlp(ue_in, training=training)      # [B*U*K, hidden]
        score = self.out(ue_h)                            # [B*U*K,1]
        score = tf.reshape(score, [B, U, K])              # [B,U,K]
        return score


def topk_mask(scores: tf.Tensor, L: int) -> tf.Tensor:
    """
    scores: [B,U,K]
    回傳 x: [B,U,K] {0,1}，每個 (B,k) 取 Top-L UE
    """
    L = int(L)
    B = tf.shape(scores)[0]
    U = tf.shape(scores)[1]
    K = tf.shape(scores)[2]

    x = tf.zeros([B, U, K], dtype=tf.float32)
    # 用 python loop 方便閱讀（主規模 K<=32 時很OK）
    for k in range(scores.shape[2]):  # 需要 K 靜態，通常你的 K 固定
        top = tf.math.top_k(scores[:, :, k], k=L).indices  # [B,L]
        # scatter: 對每個 batch 把 top indices 設成 1
        for b in range(scores.shape[0] if scores.shape[0] is not None else 1):
            idx = top[b]
            updates = tf.ones([L], tf.float32)
            x = tf.tensor_scatter_nd_update(
                x,
                indices=tf.stack([tf.fill([L], b), idx, tf.fill([L], k)], axis=1),
                updates=updates
            )
    return x
