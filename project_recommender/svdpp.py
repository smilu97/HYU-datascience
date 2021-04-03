import numpy as np
import tensorflow as tf

class SVDPPModel(tf.keras.Model):
    def __init__(self, num_users, num_items, mu, k, reg_lambda):
        super().__init__(self)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_size = k
        self.mu = mu
        self.reg_lambda = reg_lambda

        self.inference_sizes = []

        self.user_embedding = tf.keras.layers.Embedding(self.num_users, self.embed_size, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='uniform')
        self.item_embedding = tf.keras.layers.Embedding(self.num_items, self.embed_size, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='uniform')
        self.user_bias = tf.keras.layers.Embedding(self.num_users, 1, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='zeros')
        self.item_bias = tf.keras.layers.Embedding(self.num_items, 1, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='zeros')
        self.user_reshape = tf.keras.layers.Reshape((self.embed_size,))
        self.item_reshape = tf.keras.layers.Reshape((self.embed_size,))
        self.dot = tf.keras.layers.Dot(axes=(1, 1))
        self.user_bias_reshape = tf.keras.layers.Reshape((1,))
        self.item_bias_reshape = tf.keras.layers.Reshape((1,))
        self.concatenate = tf.keras.layers.Concatenate(axis=1)

        self.user_inference = tf.keras.Sequential()
        for sz in self.inference_sizes:
            self.user_inference.add(tf.keras.layers.Dense(sz))

        self.item_inference = tf.keras.Sequential()
        for sz in self.inference_sizes:
            self.item_inference.add(tf.keras.layers.Dense(sz))

    def call(self, inputs):
        user_input = inputs[0]
        item_input = inputs[1]
        
        p = self.user_embedding(user_input)
        p = self.user_reshape(p)
        p = self.user_inference(p)
        q = self.item_embedding(item_input)
        q = self.item_reshape(q)
        q = self.item_inference(q)
        pq = self.dot([p, q])
        bu = self.user_bias(user_input)
        bu = self.user_bias_reshape(bu)
        bi = self.item_bias(item_input)
        bi = self.item_bias_reshape(bi)

        return self.mu + pq + bu + bi

