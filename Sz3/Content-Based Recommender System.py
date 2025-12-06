from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit(self, products_df):
        product_features = products_df['category'] + " " + products_df['brand'] + " " + products_df['description']
        self.tfidf_matrix = self.vectorizer.fit_transform(product_features)
        self.product_ids = products_df['product_id'].values

    def recommend(self, product_id, top_n=5):
        idx = list(self.product_ids).index(product_id)
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        similar_indices = sim_scores.argsort()[-top_n - 1:-1][::-1]
        return self.product_ids[similar_indices]