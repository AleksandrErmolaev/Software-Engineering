from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class ItemBasedCF:
    def __init__(self, k_neighbors=10):
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k_neighbors)

    def fit(self, user_item_matrix):
        self.user_item_matrix = csr_matrix(user_item_matrix)
        self.knn.fit(self.user_item_matrix.T)  # Транспонируем для item-item

    def recommend(self, item_id, top_n=5):
        distances, indices = self.knn.kneighbors(
            self.user_item_matrix[:, item_id].T,
            n_neighbors=top_n + 1
        )
        similar_items = indices.flatten()[1:]  # Исключаем сам товар
        return similar_items