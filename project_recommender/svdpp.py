import numpy as np
import tensorflow as tf

from attention import Attention

# ref: https://arxiv.org/pdf/1803.03502.pdf#Attentive-Graph-based-CF-model
# ref: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.1951&rep=rep1&type=pdf

class SVDPPModel(tf.keras.Model):
    '''
    Attentive timeSVD++
    '''

    def __init__(self,
        num_users,
        num_items,
        mu,
        k,
        reg_lambda,
        sz_related,
        n_time_bins,
        use_yp_attention=True,
        use_xq_attention=False,
        use_x=True,
        use_y=True,
        use_bti=True,
        use_btu=True
    ):
        super().__init__(self)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_size = k
        self.mu = mu
        self.reg_lambda = reg_lambda
        self.sz_related = sz_related
        self.n_time_bins = n_time_bins
        self.use_yp_attention = use_yp_attention
        self.use_xq_attention = use_xq_attention
        self.use_x = use_x
        self.use_y = use_y
        self.use_bti = use_bti
        self.use_btu = use_btu


        '''
        Embeddings
        '''

        self.user_embedding = tf.keras.layers.Embedding(self.num_users, self.embed_size, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='uniform')
        self.user_x_embedding = tf.keras.layers.Embedding(self.num_users + 1, self.embed_size, input_length=sz_related,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='uniform',
            mask_zero=True)
        self.user_tu_embedding = tf.keras.layers.Embedding(self.num_users, 1, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='uniform')
        self.user_alpha_embedding = tf.keras.layers.Embedding(self.num_users, 1, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='uniform')
        self.item_embedding = tf.keras.layers.Embedding(self.num_items, self.embed_size, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='uniform')
        self.item_y_embedding = tf.keras.layers.Embedding(self.num_items + 1, self.embed_size, input_length=sz_related,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer='uniform',
            mask_zero=True)
        self.user_bias = tf.keras.layers.Embedding(self.num_users, 1, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer=tf.keras.initializers.Constant(0.0))
        self.item_bias = tf.keras.layers.Embedding(self.num_items, 1, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer=tf.keras.initializers.Constant(0.0))
        self.item_time_bin_bias = tf.keras.layers.Embedding(self.num_items * self.n_time_bins, 1, input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.reg_lambda),
            embeddings_initializer=tf.keras.initializers.Constant(0.0))
        self.yp_attention = Attention(k=self.embed_size)
        self.xq_attention = Attention(k=self.embed_size)

        '''
        Add non-linearity on hidden factors
        '''

        # self.inference_sizes = []
        # self.user_inference = tf.keras.Sequential()
        # for sz in self.inference_sizes:
        #     self.user_inference.add(tf.keras.layers.Dense(sz, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(self.reg_lambda)))

        # self.item_inference = tf.keras.Sequential()
        # for sz in self.inference_sizes:
        #     self.item_inference.add(tf.keras.layers.Dense(sz, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(self.reg_lambda)))

    def call(self, inputs):
        user_input, item_input, user_times, user_relateds, item_relateds, item_time_bins, u_time_means = inputs
        
        p = self.user_embedding(user_input)
        p = tf.reshape(p, (-1, self.embed_size,))
        # p = self.user_inference(p)

        if self.use_y:
            y = self.item_y_embedding(user_relateds)
            if self.use_yp_attention:
                y = self.yp_attention(y, p)
            else:
                y = tf.reduce_sum(y, axis=1)
                y = y / tf.maximum(1.0, tf.reshape(tf.sqrt(tf.math.count_nonzero(user_relateds, 1, dtype=tf.float32)), (-1, 1)))
        else:
            y = 0

        q = self.item_embedding(item_input)
        q = tf.reshape(q, (-1, self.embed_size,))
        # q = self.item_inference(q)

        if self.use_x:
            x = self.user_x_embedding(item_relateds)
            if self.use_xq_attention:
                x = self.xq_attention(x, q)
            else:
                x = tf.reduce_sum(x, axis=1)
                x = x / tf.maximum(1.0, tf.reshape(tf.sqrt(tf.math.count_nonzero(item_relateds, 1, dtype=tf.float32)), (-1, 1)))
        else:
            x = 0

        pq = tf.reduce_sum((p + y) * (q + x), axis=-1)

        bu = self.user_bias(user_input)
        bu = tf.reshape(bu, (-1,))

        if self.use_btu:
            tu = self.user_tu_embedding(user_input)
            tu = tf.reshape(tu, (-1,))
            dt = tf.reshape(user_times, (-1,)) - tu
            dev_ut = tf.sign(dt) * tf.pow(tf.abs(dt), 0.4)

            alpha_u = self.user_alpha_embedding(user_input)
            alpha_u = tf.reshape(alpha_u, (-1,))
            btu = alpha_u * dev_ut
        else:
            btu = 0

        bi = self.item_bias(item_input)
        bi = tf.reshape(bi, (-1,))
        
        if self.use_bti:
            bti = self.item_time_bin_bias(item_input * self.n_time_bins + item_time_bins)
            bti = tf.reshape(bti, (-1,))
        else:
            bti = 0

        return self.mu + pq + bu + btu + bi + bti

