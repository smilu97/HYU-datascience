# author: smilu97
# created at: 21-06-11 16:19

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OrdinalEncoder
from model import SVDPPModel

col_user = 0
col_item = 1
col_rating = 2
col_time = 3

class SVDtppAttRecommender:
    def __init__(self,
    min_rating=1.0,
    max_rating=5.0,
    feature_dim=12,
    lr=0.001,
    reg_lambda=0.0,
    sz_related=30,
    n_time_bins=30):

        '''
        Construct new recommender

        :param min_rating: minimum value of rating
        :param max_rating: maximum value of rating
        :param feature_dim: the size of dimensions for (user, item) representation vector
        :param lr: learning rate
        :param reg_lambda: weight regularization factor
        :param sz_related: the size of array for related (users, items)
        :param n_time_bins: the number of bins for discretizing times
        '''

        self.min_rating = min_rating
        self.max_rating = max_rating
        self.feature_dim = feature_dim
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.sz_related = sz_related
        self.n_time_bins = n_time_bins

    def fit(self, observation, batch_size=2000, epochs=0, validation_data=None):
        '''
        Build data, model, and Fit

        :param observation: numpy array which has shape (-1, 4).
            each columns must represent (user_id, item_id, time, rating)
        '''

        assert(len(observation.shape) == 2)
        assert(observation.shape[-1] == 4)

        self.observation = observation

        self.data = self._build_data()
        self.model = self._build_model()

        users = self.data[:, col_user] + 1
        items = self.data[:, col_item] + 1
        user_times = self.user_times
        user_relateds = self.user_relateds[self.data[:, col_user]]
        item_relateds = self.item_relateds[self.data[:, col_item]]
        item_time_bins = np.int64(np.minimum(self.item_times * self.n_time_bins, self.n_time_bins - 1))
        ratings = self.data[:, col_rating]
        X = [users, items, user_times, item_time_bins, user_relateds, item_relateds]
        Y = ratings

        if validation_data is not None:
            validation_data_X = self._preprocess_prediction_data(validation_data)
            validation_data_Y = validation_data[:, col_rating]
            validation_data = (validation_data_X, validation_data_Y)

        epoch = 0
        prev_val_loss = 1000000000000000000
        while epochs == 0 or epoch < epochs:
            history = self.model.fit(X, Y,
                batch_size=batch_size,
                epochs=1,
                validation_data=validation_data)
            val_loss = history.history['val_loss'][0]
            if epochs == 0 and val_loss > prev_val_loss:
                break
            prev_val_loss = val_loss
            epoch += 1
    
    def predict(self, users, items, times):
        n = len(users)
        data = np.empty((n, 4))
        data[:, col_user] = users
        data[:, col_item] = items
        data[:, col_time] = times
        X = self._preprocess_prediction_data(data)
        return self.model.predict(X)
    
    def _preprocess_prediction_data(self, data):
        n = data.shape[0]
        rel_pad = np.zeros(self.sz_related)
        users = np.empty(n, dtype=np.int64)
        items = np.empty(n, dtype=np.int64)
        user_times = np.empty(n, dtype=np.float32)
        item_time_bins = np.empty(n, dtype=np.int64)
        user_relateds = np.empty((n, self.sz_related))
        item_relateds = np.empty((n, self.sz_related))
        for i, row in enumerate(data):
            user = row[col_user]
            item = row[col_item]
            time = row[col_time]

            user_exist = user in self.users
            item_exist = item in self.items

            user = int(self.user_encoder.transform([[user]])[0, 0] if user_exist else -1)
            item = int(self.item_encoder.transform([[item]])[0, 0] if item_exist else -1)

            def norm(v, m, M):
                return (v-m)/(M-m) if m != M else 0.0

            user_times[i] = norm(time, self.user_min_times[user], self.user_max_times[user]) if user_exist else 0.0
            item_time = norm(time, self.item_min_times[item], self.item_max_times[item]) if item_exist else 0.0
            item_time_bins[i] = max(0, min(self.n_time_bins - 1, int(item_time * self.n_time_bins)))

            user_relateds[i, :] = self.user_relateds[user] if user_exist else rel_pad
            item_relateds[i, :] = self.item_relateds[item] if item_exist else rel_pad

            users[i] = user + 1
            items[i] = item + 1

        X = [users, items, user_times, item_time_bins, user_relateds, item_relateds]

        return X

    def _build_data(self):
        users, items = self._encode_user_item()
        data = np.array(self.observation)
        data[:, 0] = users.flatten()
        data[:, 1] = items.flatten()
        
        self.mu = np.mean(data[:, col_rating])

        user_times = self._normalize_time(data, col_user, self.num_users)
        item_times = self._normalize_time(data, col_item, self.num_items)
        self.user_times = user_times[0]
        self.user_min_times = user_times[1]
        self.user_max_times = user_times[2]
        self.item_times = item_times[0]
        self.item_min_times = item_times[1]
        self.item_max_times = item_times[2]

        self._build_relateds(data)

        return data
    
    def _build_relateds(self, data):
        n = data.shape[0]
        user_relateds = [list() for _ in range(self.num_users)]
        item_relateds = [list() for _ in range(self.num_items)]

        for user, item, _, _ in data:
            user_relateds[user].append(item + 1)
            item_relateds[item].append(user + 1)
        
        for i in range(self.num_users):
            user_relateds[i] = np.unique(user_relateds[i])
        for i in range(self.num_items):
            item_relateds[i] = np.unique(item_relateds[i])
        
        self.user_relateds = np.zeros((self.num_users, self.sz_related), dtype=np.int64)
        self.item_relateds = np.zeros((self.num_items, self.sz_related), dtype=np.int64)

        for index, row in enumerate(self.user_relateds):
            relateds = user_relateds[index]
            m = min(self.sz_related, len(relateds))
            row[:m] = np.array(relateds[:m])

        for index, row in enumerate(self.item_relateds):
            relateds = item_relateds[index]
            m = min(self.sz_related, len(relateds))
            row[:m] = relateds[:m]

    @staticmethod
    def _normalize_time(data, by, num_by):
        '''
        normalize time by user
        '''

        inf = 0x3f3f3f3f
        min_times = np.full(num_by, inf)
        max_times = np.zeros(num_by)
        times = np.array(data[:, col_time], dtype=np.float32)

        for row, time in zip(data, times):
            pivot = row[by]
            min_times[pivot] = min(min_times[pivot], time)
            max_times[pivot] = max(max_times[pivot], time)
        
        for i, row in enumerate(data):
            pivot = row[by]
            min_time = min_times[pivot]
            max_time = max_times[pivot]
            interval = 1.0 if min_time == max_time else max_time - min_time
            times[i] = (times[i] - min_time) / interval
        
        return times, min_times, max_times

    def _encode_user_item(self):
        ui = self.observation[:, :2]
        self.user_encoder = OrdinalEncoder()
        self.item_encoder = OrdinalEncoder()
        users = self.user_encoder.fit_transform(self.observation[:, col_user:col_user+1])
        items = self.item_encoder.fit_transform(self.observation[:, col_item:col_item+1])
        self.num_users = self.user_encoder.categories_[0].shape[0]
        self.num_items = self.item_encoder.categories_[0].shape[0]
        self.users = set(self.user_encoder.categories_[0])
        self.items = set(self.item_encoder.categories_[0])
        return users, items
    
    def _build_model(self):
        model = SVDPPModel(
            num_users=self.num_users,
            num_items=self.num_items,
            mu=self.mu,
            k=self.feature_dim,
            reg_lambda=self.reg_lambda,
            sz_related=self.sz_related,
            n_time_bins=self.n_time_bins,
            use_yp_attention=True,
            use_xq_attention=False,
            use_x=True,
            use_y=True,
            use_bti=True,
            use_btu=True
        )

        use_adam = True

        if use_adam:
            self.optimizer = tf.keras.optimizers.Adam(
                lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        else:
            self.optimizer = tf.keras.optimizers.RMSprop(
                lr=self.lr, rho=0.9, epsilon=None, decay=0.0)

        model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=[])

        return model
