# author: smilu97
# created at: 21-05-27 18:15

# ref: https://arxiv.org/pdf/1803.03502.pdf#Attentive-Graph-based-CF-model

import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, k, element_wise=True):
        super(Attention, self).__init__()
        self.k = k
        att_out = k if element_wise else 1

        w_init = tf.random_normal_initializer()
        self.att_w = tf.Variable(
            initial_value=w_init(shape=(2*k, att_out), dtype=tf.float32),
            trainable=True,
        )
        self.att_b = tf.Variable(
            initial_value=w_init(shape=(att_out,), dtype=tf.float32),
            trainable=True,
        )
    
    def call(self, y, p):
        k = self.k

        p = tf.reshape(p, (-1, 1, k))
        p = tf.repeat(p, y.shape[1], axis=1)
        yp = tf.concat([y, p], axis=-1)
        yp = tf.matmul(yp, self.att_w) + self.att_b
        yp = tf.nn.relu(yp)
        yp = tf.nn.softmax(yp, axis=-1)

        return tf.reduce_sum(y * yp, axis=1)
