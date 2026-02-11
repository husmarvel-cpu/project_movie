import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_rating_distribution(df: pd.DataFrame, column='vote_average'):
    """Гистограмма распределения рейтингов."""
    plt.figure(figsize=(10,6))
    sns.histplot(df[column].dropna(), bins=50, kde=True)
    plt.title('Распределение рейтингов фильмов')
    plt.xlabel('Рейтинг')
    plt.ylabel('Количество')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_budget_revenue_scatter(df: pd.DataFrame):
    """Диаграмма рассеяния бюджет vs выручка."""
    if 'budget' in df.columns and 'revenue' in df.columns:
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=df, x='budget', y='revenue', alpha=0.5, hue='vote_average')
        plt.title('Бюджет vs Выручка')
        plt.xlabel('Бюджет ($)')
        plt.ylabel('Выручка ($)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()

def plot_top_genres(df: pd.DataFrame, top_n=15):
    """Топ жанров."""
    if 'genres_parsed' not in df.columns:
        return
    all_genres = []
    for g in df['genres_parsed'].dropna():
        all_genres.extend(g)
    genre_counts = pd.Series(all_genres).value_counts().head(top_n)
    plt.figure(figsize=(12,6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title(f'Топ-{top_n} жанров')
    plt.xlabel('Количество фильмов')
    plt.tight_layout()
    plt.show()