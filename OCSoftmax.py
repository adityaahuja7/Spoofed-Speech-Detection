import tensorflow as tf
import numpy as np 



import tensorflow as tf

class OCSoftmax(tf.keras.layers.Layer):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = self.add_weight(shape=(1, self.feat_dim),
                                      initializer='random_normal',
                                      trainable=True)
        self.softplus = tf.keras.layers.Activation('softplus')

    def call(self, x, labels):
        """
        Args:
            x: Feature matrix with shape (batch_size, feat_dim).
            labels: Ground truth labels with shape (batch_size).
        """
        w = tf.math.l2_normalize(self.center, axis=1)
        x = tf.math.l2_normalize(x, axis=1)

        scores = tf.matmul(x, tf.transpose(w))
        output_scores = tf.identity(scores)

        scores = tf.where(labels == 0, self.r_real - scores, scores - self.r_fake)

        loss = tf.reduce_mean(self.softplus(self.alpha * scores))

        return loss, output_scores
    