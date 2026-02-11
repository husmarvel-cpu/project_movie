import pandas as pd
import numpy as np
import json
import ast
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TMDBDataLoader:

    def __init__(self, data_path: str = "data/raw/tmdb_5000_movies.csv"):
        self.data_path = data_path
        self.movies_df = None

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Загрузка данных из {self.data_path}")

        self.movies_df = pd.read_csv(self.data_path)

        # Парсинг JSON-полей
        json_columns = ['genres', 'keywords', 'production_companies',
                        'production_countries', 'spoken_languages']

        for column in json_columns:
            if column in self.movies_df.columns:
                self.movies_df[column] = self.movies_df[column].apply(
                    lambda x: self._parse_json_field(x) if pd.notna(x) else []
                )

        logger.info(f"Загружено {len(self.movies_df)} фильмов")
        return self.movies_df

    def _parse_json_field(self, json_str: str) -> List[Dict]:
        try:
            # Если это строка JSON
            if isinstance(json_str, str):
                data = ast.literal_eval(json_str)
                # Извлекаем значения 'name' из каждого словаря
                if isinstance(data, list):
                    return [item.get('name', '') for item in data]
            return []
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Ошибка парсинга JSON: {e}")
            return []

    def get_basic_stats(self) -> Dict:
        if self.movies_df is None:
            self.load_data()

        stats = {
            'total_movies': len(self.movies_df),
            'years_range': f"{self.movies_df['release_date'].min()} - {self.movies_df['release_date'].max()}",
            'total_budget': self.movies_df['budget'].sum(),
            'total_revenue': self.movies_df['revenue'].sum(),
            'avg_rating': self.movies_df['vote_average'].mean(),
            'unique_genres': len(self._get_all_genres())
        }
        return stats

    def _get_all_genres(self) -> List[str]:
        all_genres = []
        for genres_list in self.movies_df['genres']:
            all_genres.extend(genres_list)
        return list(set(all_genres))