import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class HybridMovieRecommender:
    """Гибридная рекомендательная система на основе контента и популярности."""

    def __init__(self, n_recommendations: int = 10):
        self.n_recommendations = n_recommendations
        self.movies_df = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.knn_model = None
        self.scaler = StandardScaler()

    def fit(self, movies_df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
        """Обучает модель на матрице признаков."""
        self.movies_df = movies_df.copy()

        if feature_cols is None:
            feature_cols = self._auto_select_features()

        # Создаём матрицу признаков
        self.feature_matrix = self._build_feature_matrix(feature_cols)

        # Вычисляем косинусное сходство
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

        # Обучаем KNN для быстрого поиска
        self.knn_model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
        self.knn_model.fit(self.feature_matrix)

        logger.info(f"Модель обучена на {len(self.movies_df)} фильмах")
        return self

    def _auto_select_features(self) -> List[str]:
        """Автоматически отбирает числовые признаки и one-hot жанры."""
        numeric_cols = self.movies_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['id', 'vote_average', 'vote_count', 'revenue', 'budget',
                   'profit', 'log_budget', 'log_revenue']
        feature_cols = [c for c in numeric_cols if c not in exclude]

        # Добавляем one-hot жанры
        genre_cols = [c for c in self.movies_df.columns if c.startswith('genre_')]
        feature_cols.extend(genre_cols[:15])  # ограничим для скорости

        return feature_cols[:30]  # не более 30 признаков

    def _build_feature_matrix(self, feature_cols: List[str]) -> np.ndarray:
        """Создаёт нормализованную матрицу признаков."""
        features = self.movies_df[feature_cols].fillna(0)
        return self.scaler.fit_transform(features)

    def recommend_by_content(self, movie_id: int, n: int = None) -> pd.DataFrame:
        """Рекомендации на основе сходства с заданным фильмом."""
        if n is None:
            n = self.n_recommendations

        idx = self.movies_df[self.movies_df['id'] == movie_id].index
        if len(idx) == 0:
            raise ValueError(f"Фильм с id {movie_id} не найден")
        idx = idx[0]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n + 1]  # исключаем сам фильм

        movie_indices = [i[0] for i in sim_scores]
        similarities = [i[1] for i in sim_scores]

        result = self.movies_df.iloc[movie_indices][['id', 'title', 'vote_average', 'release_year']].copy()
        result['similarity_score'] = similarities

        # Добавляем жанры, если есть
        if 'genres_parsed' in self.movies_df.columns:
            result['genres'] = self.movies_df.iloc[movie_indices]['genres_parsed'].apply(lambda x: x[:3] if x else [])

        return result

    def recommend_by_popularity(self, genre: Optional[str] = None,
                                year_range: Optional[Tuple[int, int]] = None,
                                n: int = None) -> pd.DataFrame:
        """Рекомендации по популярности (взвешенный рейтинг)."""
        if n is None:
            n = self.n_recommendations

        df = self.movies_df.copy()

        # Фильтр по жанру
        if genre and 'genres_parsed' in df.columns:
            df = df[df['genres_parsed'].apply(lambda x: genre in x if x else False)]

        # Фильтр по годам
        if year_range and 'release_year' in df.columns:
            df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]

        # Сортировка по взвешенному рейтингу или vote_average
        if 'weighted_rating' in df.columns:
            df = df.sort_values('weighted_rating', ascending=False)
        else:
            df['score'] = df['vote_average'] * np.log1p(df['vote_count'])
            df = df.sort_values('score', ascending=False)

        return df.head(n)[['id', 'title', 'vote_average', 'vote_count', 'release_year', 'genres_parsed']]

    def recommend_by_preferences(self, preferences: Dict) -> pd.DataFrame:
        """Персонализированные рекомендации на основе жанров, рейтинга и годов."""
        df = self.movies_df.copy()

        # Фильтр по жанрам
        if 'genres' in preferences and preferences['genres'] and 'genres_parsed' in df.columns:
            genre_list = preferences['genres']
            df = df[df['genres_parsed'].apply(
                lambda x: any(g in x for g in genre_list) if x else False
            )]

        # Фильтр по минимальному рейтингу
        if 'min_rating' in preferences:
            df = df[df['vote_average'] >= preferences['min_rating']]

        # Фильтр по годам
        if 'year_range' in preferences:
            ymin, ymax = preferences['year_range']
            df = df[(df['release_year'] >= ymin) & (df['release_year'] <= ymax)]

        if len(df) == 0:
            logger.warning("Нет фильмов, соответствующих критериям")
            return pd.DataFrame()

        # Сортируем
        if 'weighted_rating' in df.columns:
            df = df.sort_values('weighted_rating', ascending=False)
        else:
            df['score'] = df['vote_average'] * np.log1p(df['vote_count'])
            df = df.sort_values('score', ascending=False)

        return df.head(self.n_recommendations)[['id', 'title', 'vote_average', 'release_year', 'genres_parsed']]

    def recommend_hybrid(self, movie_id: Optional[int] = None,
                         preferences: Optional[Dict] = None) -> pd.DataFrame:
        """Гибридные рекомендации: content-based + популярность."""
        if movie_id:
            # Похожие фильмы
            content_recs = self.recommend_by_content(movie_id, n=20)
            # Добавляем популярность как дополнительный вес
            if 'weighted_rating' in self.movies_df.columns:
                ratings = self.movies_df.set_index('id')['weighted_rating']
                content_recs['popularity_score'] = content_recs['id'].map(ratings)
                content_recs['final_score'] = (content_recs['similarity_score'] * 0.7 +
                                               content_recs['popularity_score'] * 0.3)
                content_recs = content_recs.sort_values('final_score', ascending=False)
            return content_recs.head(self.n_recommendations)
        elif preferences:
            return self.recommend_by_preferences(preferences)
        else:
            return self.recommend_by_popularity()