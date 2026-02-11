import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re


class FeatureEngineer:

    def __init__(self):
        self.mlb_genres = MultiLabelBinarizer()
        self.mlb_keywords = MultiLabelBinarizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = MinMaxScaler()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = df.copy()

        features_df = self._create_financial_features(features_df)

        features_df = self._create_time_features(features_df)

        features_df = self._encode_genres(features_df)

        features_df = self._create_text_features(features_df)

        features_df = self._create_popularity_features(features_df)

        features_df = self._create_advanced_metrics(features_df)

        return features_df

    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['profit'] = df['revenue'] - df['budget']
        df['roi'] = np.where(df['budget'] > 0, df['profit'] / df['budget'], 0)
        df['budget_to_revenue_ratio'] = np.where(df['revenue'] > 0, df['budget'] / df['revenue'], 0)
        df['is_profitable'] = (df['profit'] > 0).astype(int)

        financial_cols = ['budget', 'revenue', 'profit']
        for col in financial_cols:
            df[f'log_{col}'] = np.log1p(df[col])

        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day_of_week'] = df['release_date'].dt.dayofweek
        df['release_season'] = df['release_date'].dt.month % 12 // 3 + 1

        # Возраст фильма
        current_year = pd.Timestamp.now().year
        df['movie_age'] = current_year - df['release_year']

        return df

    def _encode_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        genres_matrix = self.mlb_genres.fit_transform(df['genres'])
        genres_df = pd.DataFrame(
            genres_matrix,
            columns=[f"genre_{col}" for col in self.mlb_genres.classes_],
            index=df.index
        )

        # Добавляем количество жанров
        df['num_genres'] = df['genres'].apply(len)

        return pd.concat([df, genres_df], axis=1)

    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        overview_clean = df['overview'].fillna('').apply(self._clean_text)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(overview_clean)

        svd = TruncatedSVD(n_components=20, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)

        for i in range(tfidf_reduced.shape[1]):
            df[f'tfidf_component_{i}'] = tfidf_reduced[:, i]

        df['overview_length'] = df['overview'].fillna('').apply(len)
        df['title_length'] = df['title'].apply(len)

        df['has_tagline'] = df['tagline'].notna().astype(int)

        return df

    def _create_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(0.75)

        def weighted_rating(row):
            v = row['vote_count']
            R = row['vote_average']
            return (v / (v + m) * R) + (m / (v + m) * C)

        df['weighted_rating'] = df.apply(weighted_rating, axis=1)

        df['norm_popularity'] = self.scaler.fit_transform(df[['popularity']])
        df['norm_vote_count'] = self.scaler.fit_transform(df[['vote_count']])

        return df

    def _create_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df['budget_efficiency'] = np.where(
            df['budget'] > 0,
            df['revenue'] / df['budget'],
            0
        )

        df['genre_density'] = df['num_genres'] / len(self.mlb_genres.classes_)

        df['success_score'] = (
                df['weighted_rating'] * 0.4 +
                df['norm_popularity'] * 0.3 +
                df['roi'] * 0.3
        )

        return df

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text