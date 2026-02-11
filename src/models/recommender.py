import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pickle
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridMovieRecommender:

    def __init__(self, n_recommendations: int = 10):
        self.n_recommendations = n_recommendations
        self.movies_df = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.knn_model = None

    def fit(self, movies_df: pd.DataFrame, feature_columns: List[str] = None):
        self.movies_df = movies_df.copy()

        if feature_columns is None:
            feature_columns = self._select_feature_columns()

        self.feature_matrix = self._create_feature_matrix(feature_columns)

        self.similarity_matrix = cosine_similarity(self.feature_matrix)

        self.knn_model = NearestNeighbors(
            n_neighbors=50,
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(self.feature_matrix)

        logger.info(f"Модель обучена на {len(self.movies_df)} фильмах")
        return self

    def _select_feature_columns(self) -> List[str]:
        numeric_cols = self.movies_df.select_dtypes(include=[np.number]).columns.tolist()

        exclude = ['id', 'vote_average', 'vote_count', 'revenue', 'budget']
        feature_cols = [col for col in numeric_cols if col not in exclude]

        genre_cols = [col for col in self.movies_df.columns if col.startswith('genre_')]
        feature_cols.extend(genre_cols[:20])  # Ограничиваем количество

        return feature_cols

    def _create_feature_matrix(self, feature_columns: List[str]) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        features = self.movies_df[feature_columns].fillna(0)

        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(features)

        return feature_matrix

    def recommend_by_content(self, movie_id: int, n: int = None) -> pd.DataFrame:
        if n is None:
            n = self.n_recommendations

        movie_idx = self.movies_df[self.movies_df['id'] == movie_id].index[0]

        similarities = self.similarity_matrix[movie_idx]

        similar_indices = np.argsort(similarities)[::-1][1:n + 1]

        recommendations = []
        for idx in similar_indices:
            movie_data = self.movies_df.iloc[idx]
            recommendations.append({
                'id': movie_data['id'],
                'title': movie_data['title'],
                'similarity_score': similarities[idx],
                'genres': movie_data['genres'][:3],
                'vote_average': movie_data['vote_average'],
                'release_year': movie_data.get('release_year', 'N/A')
            })

        return pd.DataFrame(recommendations)

    def recommend_by_popularity(self, genre: str = None, year_range: Tuple = None) -> pd.DataFrame:
        df = self.movies_df.copy()

        if genre:
            df = df[df['genres'].apply(lambda x: genre in x)]

        if year_range:
            df = df[(df['release_year'] >= year_range[0]) &
                    (df['release_year'] <= year_range[1])]

        if 'weighted_rating' in df.columns:
            df = df.sort_values('weighted_rating', ascending=False)
        else:
            df['score'] = df['vote_average'] * np.log1p(df['vote_count'])
            df = df.sort_values('score', ascending=False)

        return df.head(self.n_recommendations)[[
            'id', 'title', 'vote_average', 'vote_count',
            'release_year', 'genres'
        ]]

    def recommend_hybrid(self, movie_id: int = None,
                         user_preferences: Dict = None) -> pd.DataFrame:
        if movie_id:
            content_recs = self.recommend_by_content(movie_id, n=20)

            if user_preferences:
                content_recs = self._filter_by_preferences(content_recs, user_preferences)

            if 'weighted_rating' in self.movies_df.columns:
                ratings = self.movies_df.set_index('id')['weighted_rating']
                content_recs['weighted_score'] = content_recs['id'].map(ratings)
                content_recs['final_score'] = (
                        content_recs['similarity_score'] * 0.6 +
                        content_recs['weighted_score'] * 0.4
                )
                content_recs = content_recs.sort_values('final_score', ascending=False)

            return content_recs.head(self.n_recommendations)

        elif user_preferences:
            return self.recommend_by_preferences(user_preferences)

        else:
            return self.recommend_by_popularity()

    def _filter_by_preferences(self, df: pd.DataFrame,
                               preferences: Dict) -> pd.DataFrame:
        filtered_df = df.copy()

        if 'genres' in preferences:
            genre_filter = preferences['genres']
            filtered_df = filtered_df[
                filtered_df['genres'].apply(
                    lambda x: any(genre in x for genre in genre_filter)
                )
            ]

        if 'min_rating' in preferences:
            filtered_df = filtered_df[filtered_df['vote_average'] >= preferences['min_rating']]

        if 'year_range' in preferences:
            min_year, max_year = preferences['year_range']
            filtered_df = filtered_df[
                (filtered_df['release_year'] >= min_year) &
                (filtered_df['release_year'] <= max_year)
                ]

        return filtered_df

    def recommend_by_preferences(self, preferences: Dict) -> pd.DataFrame:
        df = self.movies_df.copy()

        df = self._filter_by_preferences(df, preferences)

        if len(df) == 0:
            logger.warning("No movies match preferences, relaxing criteria")
            if 'min_rating' in preferences:
                preferences['min_rating'] = max(6.0, preferences['min_rating'] - 1)
                df = self.movies_df.copy()
                df = self._filter_by_preferences(df, preferences)

        if 'weighted_rating' in df.columns:
            df = df.sort_values(['weighted_rating', 'popularity'], ascending=False)
        else:
            df['score'] = df['vote_average'] * np.log1p(df['vote_count'])
            df = df.sort_values(['score', 'popularity'], ascending=False)

        return df.head(self.n_recommendations)[[
            'id', 'title', 'vote_average', 'vote_count',
            'release_year', 'genres'
        ]]

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'movies_df': self.movies_df,
                'feature_matrix': self.feature_matrix,
                'similarity_matrix': self.similarity_matrix,
                'knn_model': self.knn_model
            }, f)
        logger.info(f"Модель сохранена в {path}")

    @classmethod
    def load_model(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        recommender = cls()
        recommender.movies_df = data['movies_df']
        recommender.feature_matrix = data['feature_matrix']
        recommender.similarity_matrix = data['similarity_matrix']
        recommender.knn_model = data['knn_model']

        logger.info(f"Модель загружена из {path}")
        return recommender