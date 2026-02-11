import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_tmdb_dataset
from src.features.engineering import FeatureEngineer
from src.models.recommender import HybridMovieRecommender

st.set_page_config(page_title="TMDB Movie Analytics", page_icon="🎬", layout="wide")


@st.cache_resource
def load_data_and_model():
    movies, credits = load_tmdb_dataset()

    # ========== УНИВЕРСАЛЬНАЯ НОРМАЛИЗАЦИЯ КОЛОНОК ==========
    # Приводим все названия колонок к нижнему регистру и заменяем пробелы на _
    movies.columns = movies.columns.str.lower().str.replace(' ', '_')

    # Если колонка называется 'budget' или 'budget' (теперь точно lower) — она есть
    # Если нет — создаём заглушку (но в TMDB она должна быть)
    if 'budget' not in movies.columns:
        movies['budget'] = 0
    if 'revenue' not in movies.columns:
        movies['revenue'] = 0
    if 'vote_average' not in movies.columns:
        movies['vote_average'] = 0
    if 'vote_count' not in movies.columns:
        movies['vote_count'] = 0
    if 'popularity' not in movies.columns:
        movies['popularity'] = 0
    if 'release_date' not in movies.columns:
        movies['release_date'] = pd.NaT

    # ========== FEATURE ENGINEERING ==========
    fe = FeatureEngineer()
    movies_fe = fe.create_features(movies)

    # ========== РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА ==========
    recommender = HybridMovieRecommender(n_recommendations=10)
    recommender.fit(movies_fe)

    return movies, movies_fe, recommender


st.title("🎬 TMDB Movie Analytics & Recommendation System")
st.markdown("---")

with st.spinner("Загрузка данных и модели..."):
    movies, movies_fe, recommender = load_data_and_model()

# Боковая навигация
page = st.sidebar.radio(
    "Навигация",
    ["📊 Обзор данных", "🔍 Аналитика", "🎯 Рекомендации", "📈 Тренды"]
)

# ========== СТРАНИЦА 1: ОБЗОР ==========
if page == "📊 Обзор данных":
    st.header("Обзор датасета TMDB")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего фильмов", len(movies))
    col2.metric("Средний рейтинг", f"{movies['vote_average'].mean():.2f}")

    # Бюджет и выручка могут быть нулевыми — это нормально
    avg_budget = movies['budget'].mean()
    avg_revenue = movies['revenue'].mean()
    col3.metric("Средний бюджет", f"${avg_budget:,.0f}" if avg_budget > 0 else "N/A")
    col4.metric("Средняя выручка", f"${avg_revenue:,.0f}" if avg_revenue > 0 else "N/A")

    st.subheader("Пример данных")
    # Показываем только существующие колонки
    display_cols = ['title', 'vote_average', 'release_date', 'budget', 'revenue']
    display_cols = [c for c in display_cols if c in movies.columns]
    st.dataframe(movies[display_cols].head(100))

    st.subheader("Распределение рейтингов")
    fig = px.histogram(movies, x='vote_average', nbins=50, title='Распределение рейтингов')
    st.plotly_chart(fig, use_container_width=True)

# ========== СТРАНИЦА 2: АНАЛИТИКА ==========
elif page == "🔍 Аналитика":
    st.header("Аналитика фильмов")

    # Доступные числовые колонки для графиков
    numeric_cols = movies.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c in ['budget', 'revenue', 'vote_average', 'popularity', 'release_year']]

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Ось X", numeric_cols, index=0)
        with col2:
            y_axis = st.selectbox("Ось Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

        fig = px.scatter(movies_fe, x=x_axis, y=y_axis,
                         color='vote_average', size='vote_count',
                         hover_data=['title'],
                         title=f'{y_axis} vs {x_axis}',
                         log_x=True if x_axis in ['budget', 'revenue'] else False,
                         log_y=True if y_axis in ['budget', 'revenue'] else False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Недостаточно числовых данных для графиков")

    # Топ жанров (если есть колонка genres_parsed)
    st.subheader("Топ жанров")
    if 'genres_parsed' in movies.columns:
        all_genres = []
        for g in movies['genres_parsed'].dropna():
            all_genres.extend(g)
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts().head(15)
            fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                         title='Топ-15 жанров')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Колонка с жанрами не найдена")

# ========== СТРАНИЦА 3: РЕКОМЕНДАЦИИ ==========
elif page == "🎯 Рекомендации":
    st.header("Система рекомендаций")

    rec_type = st.radio("Тип рекомендаций",
                        ["🎬 Похожие фильмы", "🔥 Популярные", "✨ Персонализированные"])

    if rec_type == "🎬 Похожие фильмы":
        if 'title' in movies.columns:
            movie_titles = movies['title'].dropna().sort_values().tolist()
            selected = st.selectbox("Выберите фильм", movie_titles)
            movie_id = movies[movies['title'] == selected]['id'].values[0]

            if st.button("Найти похожие"):
                try:
                    recs = recommender.recommend_by_content(movie_id)
                    st.subheader(f"Фильмы, похожие на «{selected}»:")
                    cols = st.columns(3)
                    for i, (_, row) in enumerate(recs.iterrows()):
                        with cols[i % 3]:
                            st.markdown(f"""
                            **{row['title']}**  
                            ⭐ {row['vote_average']:.1f}  
                            🔗 Сходство: {row['similarity_score']:.2f}
                            """)
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        else:
            st.warning("Нет данных о фильмах")

    elif rec_type == "🔥 Популярные":
        col1, col2 = st.columns(2)
        with col1:
            genres_list = []
            if 'genres_parsed' in movies.columns:
                for g in movies['genres_parsed'].dropna():
                    genres_list.extend(g)
            genres_list = sorted(set(genres_list))
            genre = st.selectbox("Жанр (опционально)", ["Все"] + genres_list if genres_list else ["Все"])
        with col2:
            year_range = st.slider("Годы", 1900, 2025, (2000, 2020))

        genre_filter = None if genre == "Все" else genre
        recs = recommender.recommend_by_popularity(genre=genre_filter, year_range=year_range)
        st.dataframe(recs)

    else:  # Персонализированные
        st.write("Укажите ваши предпочтения:")
        genres_list = []
        if 'genres_parsed' in movies.columns:
            for g in movies['genres_parsed'].dropna():
                genres_list.extend(g)
        genres_list = sorted(set(genres_list))

        if genres_list:
            selected_genres = st.multiselect("Любимые жанры", genres_list,
                                             default=genres_list[:2] if len(genres_list) >= 2 else [])
            min_rating = st.slider("Минимальный рейтинг", 0.0, 10.0, 7.0, 0.5)
            year_min = st.number_input("Минимальный год", 1900, 2025, 2000)

            if st.button("Получить рекомендации"):
                prefs = {
                    'genres': selected_genres,
                    'min_rating': min_rating,
                    'year_range': (year_min, 2025)
                }
                recs = recommender.recommend_by_preferences(prefs)
                if len(recs) > 0:
                    fig = px.bar(recs.head(10), x='title', y='vote_average',
                                 color='vote_average', title="Топ рекомендаций")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Нет фильмов по заданным критериям")
        else:
            st.info("Нет данных о жанрах для персонализированных рекомендаций")

# ========== СТРАНИЦА 4: ТРЕНДЫ ==========
else:
    st.header("Тренды киноиндустрии")
    if 'release_year' in movies_fe.columns and 'revenue' in movies_fe.columns:
        yearly = movies_fe.groupby('release_year').agg({
            'revenue': 'sum',
            'budget': 'sum',
            'vote_average': 'mean',
            'id': 'count'
        }).rename(columns={'id': 'count'}).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly['release_year'], y=yearly['revenue'],
                                 mode='lines+markers', name='Выручка', yaxis='y'))
        fig.add_trace(go.Bar(x=yearly['release_year'], y=yearly['count'],
                             name='Количество фильмов', yaxis='y2', opacity=0.3))
        fig.update_layout(
            title='Динамика по годам',
            xaxis_title='Год',
            yaxis_title='Выручка ($)',
            yaxis2=dict(title='Количество фильмов', overlaying='y', side='right')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Недостаточно данных для построения трендов")