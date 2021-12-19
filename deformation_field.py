import tensorflow as tf


class DeformationField(tf.Module):
    """
    Image-plane deformation field.

    This is an MLP with D layers and W weights per layer. Skip connections
    are defined by a list passed to the constructor.

    The input to the MLP is a 2D image-plane coordinate.
    The output is a 2D image-plane vector used to shift a camera
    ray's endpoint from the input coordinates to some other coordinates.
    """

    def __init__(self, D=6, W=128, input_ch=2, output_ch=2, skips=[3]):
        super(DeformationField, self).__init__()

        relu = tf.keras.layers.ReLU()
        dense = lambda W, act=relu: tf.keras.layers.Dense(W, activation=act)

        input_ch = int(input_ch)

        inputs_pts = tf.keras.Input(shape=input_ch)
        inputs_pts.set_shape([None, input_ch])

        outputs = inputs_pts
        for i in range(D):
            outputs = dense(W)(outputs)
            if i in skips:
                outputs = tf.concat([inputs_pts, outputs], -1)

        outputs = dense(output_ch, act=None)(outputs)

        self.model = tf.keras.Model(inputs=inputs_pts, outputs=outputs)

    def __call__(self, pts):
        return self.model(pts)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
