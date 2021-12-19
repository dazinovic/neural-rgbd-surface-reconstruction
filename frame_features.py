import tensorflow as tf


class FeatureArray(tf.Module):
    """
    Per-frame corrective latent code.
    """

    def __init__(self, num_frames, num_channels):
        super(FeatureArray, self).__init__()

        self.num_frames = num_frames
        self.num_channels = num_channels

        self.data = tf.Variable(
            tf.random.normal([num_frames, num_channels], dtype=tf.float32)
        )

    def __call__(self, ids):
        ids = tf.where(ids < self.num_frames, ids, tf.zeros_like(ids))
        return tf.gather(self.data, ids)

    def get_weights(self):
        return self.data.numpy()

    def set_weights(self, weights):
        self.data.assign(weights)
