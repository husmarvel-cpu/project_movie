import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re


class FeatureEngineer:
    """Создание признаков для фильмов."""

    def __init__(self):
        self.mlb_genres = MultiLabelBinarizer()
        self.scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(max_features=500, stop_words='english')

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Финансовые признаки
        df = self._add_financial_features(df)

        # 2. Временные признаки
        df = self._add_time_features(df)

        # 3. Жанры one-hot
        df = self._add_genre_features(df)

        # 4. Признаки из текста (overview)
        df = self._add_text_features(df)

        # 5. Популярность и рейтинг
        df = self._add_popularity_features(df)

        # 6. Сложные метрики
        df = self._add_advanced_features(df)

        return df

    def _add_financial_features(self, df):
        if 'budget' in df.columns and 'revenue' in df.columns:
            df['profit'] = df['revenue'] - df['budget']
            df['roi'] = np.where(df['budget'] > 0, df['profit'] / df['budget'], 0)
            df['log_budget'] = np.log1p(df['budget'])
            df['log_revenue'] = np.log1p(df['revenue'])
        return df

    def _add_time_features(self, df):
        if 'release_date' in df.columns:
            df['release_year'] = df['release_date'].dt.year
            df['release_month'] = df['release_date'].dt.month
            df['release_dayofweek'] = df['release_date'].dt.dayofweek
            df['release_season'] = df['release_month'] % 12 // 3 + 1
            current_year = pd.Timestamp.now().year
            df['movie_age'] = current_year - df['release_year']
        return df

    def _add_genre_features(self, df):
        if 'genres_parsed' in df.columns:
            genres_matrix = self.mlb_genres.fit_transform(df['genres_parsed'])
            genre_cols = [f"genre_{g}" for g in self.mlb_genres.classes_]
            genres_df = pd.DataFrame(genres_matrix, columns=genre_cols, index=df.index)
            df = pd.concat([df, genres_df], axis=1)
            df['num_genres'] = df['genres_parsed'].apply(len)
        return df

    def _add_text_features(self, df):
        if 'overview' in df.columns:
            overviews = df['overview'].fillna('').apply(self._clean_text)
            tfidf_mat = self.tfidf.fit_transform(overviews)
            svd = TruncatedSVD(n_components=20, random_state=42)
            tfidf_reduced = svd.fit_transform(tfidf_mat)
            for i in range(tfidf_reduced.shape[1]):
                df[f'tfidf_{i}'] = tfidf_reduced[:, i]
            df['overview_length'] = overviews.apply(len)
        return df

    def _add_popularity_features(self, df):
        if 'vote_average' in df.columns and 'vote_count' in df.columns:
            C = df['vote_average'].mean()
            m = df['vote_count'].quantile(0.75)

            def weighted_rating(row):
                v = row['vote_count']
                R = row['vote_average']
                return (v / (v + m) * R) + (m / (v + m) * C)

            df['weighted_rating'] = df.apply(weighted_rating, axis=1)
        if 'popularity' in df.columns:
            df['norm_popularity'] = self.scaler.fit_transform(df[['popularity']])
        return df

    def _add_advanced_features(self, df):
        if 'budget' in df.columns and 'revenue' in df.columns:
            df['budget_efficiency'] = np.where(df['budget'] > 0, df['revenue'] / df['budget'], 0)
        if 'num_genres' in df.columns and 'genres_parsed' in df.columns:
            total_genres = len(self.mlb_genres.classes_) if self.mlb_genres.classes_.size > 0 else 1
            df['genre_density'] = df['num_genres'] / total_genres
        return df

    @staticmethod
    def _clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text