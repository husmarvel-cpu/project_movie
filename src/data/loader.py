import pandas as pd
import numpy as np
import ast
import json
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TMDBDataLoader:
    """
    Железобетонный загрузчик TMDB.
    Ищет CSV-файл с фильмами во всех разумных местах.
    """

    def __init__(self):
        self.movies_df = None
        self.credits_df = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Загружает датасет, найдя файл с фильмами."""
        movies_path = self._find_movies_file()

        if movies_path is None:
            # Выдаём понятную ошибку с инструкцией
            self._print_debug_info()
            raise FileNotFoundError(
                "❌ Файл с фильмами не найден!\n"
                "1. Убедись, что CSV-файл с фильмами лежит в папке 'data/raw/'\n"
                "2. Название файла должно содержать 'movie' или 'movies' (регистр не важен)\n"
                "3. Если файл уже там — проверь название и перезапусти ноутбук"
            )

        logger.info(f"✅ Загружаем: {movies_path}")
        self.movies_df = pd.read_csv(movies_path)

        # Парсинг JSON-полей
        self._parse_movies()

        # Пустой датафрейм для совместимости
        self.credits_df = pd.DataFrame(columns=['movie_id', 'title', 'cast', 'crew'])

        logger.info(f"✅ Загружено фильмов: {len(self.movies_df)}")
        return self.movies_df, self.credits_df

    def _find_movies_file(self) -> Optional[Path]:
        """Ищет CSV-файл с фильмами в нескольких местах."""

        # Список возможных путей для поиска
        search_paths = self._get_search_paths()

        for path in search_paths:
            if path.exists():
                # Если это папка, ищем в ней CSV с 'movie' в имени
                if path.is_dir():
                    for file in path.glob('*.csv'):
                        if 'movie' in file.name.lower():
                            return file
                # Если это файл и он CSV — берём его
                elif path.is_file() and path.suffix.lower() == '.csv':
                    return path

        return None

    def _get_search_paths(self):
        """Генерирует все возможные пути к данным."""
        paths = []

        # 1. Абсолютный путь относительно этого файла (корень проекта)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # поднимаемся из src/data/ -> src/ -> корень
        paths.append(project_root / 'data' / 'raw')
        paths.append(project_root / 'data')

        # 2. Относительно текущей рабочей директории
        cwd = Path.cwd()
        paths.append(cwd / 'data' / 'raw')
        paths.append(cwd / 'data')
        paths.append(cwd.parent / 'data' / 'raw')  # если ноутбук в notebooks/
        paths.append(cwd.parent / 'data')

        # 3. Относительно корня диска (на всякий случай)
        paths.append(Path('C:/Users/Khusein/Desktop/project_movie/data/raw'))
        paths.append(Path('C:/Users/Khusein/Desktop/project_movie/data'))

        # 4. Текущая папка и родительские
        paths.append(Path('.'))
        paths.append(Path('..'))

        return paths

    def _print_debug_info(self):
        """Выводит отладочную информацию, чтобы помочь найти проблему."""
        print("\n" + "=" * 60)
        print("🔍 ОТЛАДКА ЗАГРУЗЧИКА ДАННЫХ")
        print("=" * 60)
        print(f"Текущая рабочая папка: {Path.cwd()}")
        print(f"Папка с кодом loader.py: {Path(__file__).resolve().parent}")
        print("\nПроверенные места:")
        for path in self._get_search_paths():
            if path.exists():
                if path.is_dir():
                    files = list(path.glob('*.csv'))
                    if files:
                        print(f"  ✅ {path} (найдены CSV: {[f.name for f in files]})")
                    else:
                        print(f"  ⚠️  {path} (нет CSV)")
                else:
                    print(f"  ✅ {path} (файл существует)")
            else:
                print(f"  ❌ {path} (не существует)")
        print("=" * 60 + "\n")

    def _parse_movies(self):
        """Парсит JSON-поля в списки названий."""
        json_cols = ['genres', 'keywords', 'production_companies',
                     'production_countries', 'spoken_languages']
        for col in json_cols:
            if col in self.movies_df.columns:
                self.movies_df[f'{col}_parsed'] = self.movies_df[col].apply(
                    self._safe_parse_json
                )

        if 'release_date' in self.movies_df.columns:
            self.movies_df['release_date'] = pd.to_datetime(
                self.movies_df['release_date'], errors='coerce'
            )

    @staticmethod
    def _safe_parse_json(x):
        if pd.isna(x):
            return []
        try:
            if isinstance(x, str):
                data = ast.literal_eval(x)
            else:
                data = x
            if isinstance(data, list):
                return [item.get('name', '') for item in data if isinstance(item, dict)]
            return []
        except:
            try:
                if isinstance(x, str):
                    data = json.loads(x)
                    if isinstance(data, list):
                        return [item.get('name', '') for item in data if isinstance(item, dict)]
                return []
            except:
                return []


def load_tmdb_dataset():
    """Удобная функция для импорта."""
    loader = TMDBDataLoader()
    return loader.load_data()