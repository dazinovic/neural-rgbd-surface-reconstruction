import tensorflow as tf


class PoseArray(tf.Module):
    """
    Per-frame camera pose correction.

    The pose correction contains 6 parameters for each pose (3 for rotation, 3 for translation).
    The rotation parameters define Euler angles which can be converted into a rotation matrix.
    """

    def __init__(self, num_frames):
        super(PoseArray, self).__init__()

        self.num_frames = num_frames
        self.num_params = 6

        self.data = tf.Variable(
            tf.zeros([self.num_frames, self.num_params], dtype=tf.float32)
        )

    def __call__(self, ids):
        return tf.gather(self.data, ids)

    def get_weights(self):
        return self.data.numpy()

    def set_weights(self, weights):
        self.data.assign(weights)

    def get_translations(self, ids):
        return tf.gather(self.data[:, 3:6], ids)

    def get_rotations(self, ids):
        return tf.gather(self.data[:, 0:3], ids)

    def get_rotation_matrices(self, ids):
        rotations = self.get_rotations(ids)  # [N_frames, 3]

        cos_alpha = tf.math.cos(rotations[:, 0])
        cos_beta = tf.math.cos(rotations[:, 1])
        cos_gamma = tf.math.cos(rotations[:, 2])
        sin_alpha = tf.math.sin(rotations[:, 0])
        sin_beta = tf.math.sin(rotations[:, 1])
        sin_gamma = tf.math.sin(rotations[:, 2])

        col1 = tf.stack([cos_alpha * cos_beta,
                         sin_alpha * cos_beta,
                         -sin_beta], -1)
        col2 = tf.stack([cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma,
                         sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma,
                         cos_beta * sin_gamma], -1)
        col3 = tf.stack([cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma,
                         sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma,
                         cos_beta * cos_gamma], -1)

        return tf.stack([col1, col2, col3], -1)

    def transform_points(self, points, ids):
        R = self.get_rotation_matrices(ids)
        t = self.get_translations(ids)

        return tf.reduce_sum(points[..., None, :] * R, -1) + t
