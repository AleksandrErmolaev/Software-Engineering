import tensorflow as tf
from tensorflow.keras import layers, Model


class TwoTowerModel(Model):
    def __init__(self, user_dim=64, item_dim=64, embedding_dim=32):
        super().__init__()
        # User Tower
        self.user_embedding = layers.Embedding(input_dim=10000, output_dim=embedding_dim)
        self.user_dense1 = layers.Dense(user_dim, activation='relu')
        self.user_dense2 = layers.Dense(embedding_dim, activation='relu')

        # Item Tower
        self.item_embedding = layers.Embedding(input_dim=5000, output_dim=embedding_dim)
        self.item_dense1 = layers.Dense(item_dim, activation='relu')
        self.item_dense2 = layers.Dense(embedding_dim, activation='relu')

    def call(self, inputs):
        user_ids, item_ids = inputs
        user_emb = self.user_embedding(user_ids)
        user_vec = self.user_dense1(user_emb)
        user_vec = self.user_dense2(user_vec)

        item_emb = self.item_embedding(item_ids)
        item_vec = self.item_dense1(item_emb)
        item_vec = self.item_dense2(item_vec)

        # Cosine similarity
        similarity = tf.reduce_sum(user_vec * item_vec, axis=1)
        return similarity