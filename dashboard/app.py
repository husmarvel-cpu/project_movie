import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.append('..')

from src.data.loader import load_tmdb_dataset
from src.features.engineering import FeatureEngineer
from src.models.recommender import HybridMovieRecommender

st.set_page_config(page_title="TMDB Movie Analytics", page_icon="🎬", layout="wide")


@st.cache_resource
def load_data_and_model():
    movies, credits = load_tmdb_dataset()
    fe = FeatureEngineer()
    movies_fe = fe.create_features(movies)
    recommender = HybridMovieRecommender(n_recommendations=10)
    recommender.fit(movies_fe)
    return movies, movies_fe, recommender


st.title("🎬 TMDB Movie Analytics & Recommendation System")
st.markdown("---")

with st.spinner("Загрузка данных и модели..."):
    movies, movies_fe, recommender = load_data_and_model()

# Сайдбар навигация
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
    col3.metric("Средний бюджет", f"${movies['budget'].mean():,.0f}")
    col4.metric("Средняя выручка", f"${movies['revenue'].mean():,.0f}")

    st.subheader("Пример данных")
    st.dataframe(movies[['title', 'vote_average', 'release_date', 'budget', 'revenue']].head(100))

    st.subheader("Распределение рейтингов")
    fig = px.histogram(movies, x='vote_average', nbins=50, title='Распределение рейтингов')
    st.plotly_chart(fig, use_container_width=True)

# ========== СТРАНИЦА 2: АНАЛИТИКА ==========
elif page == "🔍 Аналитика":
    st.header("Аналитика фильмов")

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Ось X", ['budget', 'revenue', 'vote_average', 'popularity', 'release_year'])
    with col2:
        y_axis = st.selectbox("Ось Y", ['revenue', 'vote_average', 'popularity', 'budget'])

    fig = px.scatter(movies_fe, x=x_axis, y=y_axis,
                     color='vote_average', size='vote_count',
                     hover_data=['title'],
                     title=f'{y_axis} vs {x_axis}',
                     log_x=True if x_axis in ['budget', 'revenue'] else False,
                     log_y=True if y_axis in ['budget', 'revenue'] else False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Топ жанров")
    if 'genres_parsed' in movies.columns:
        all_genres = []
        for g in movies['genres_parsed'].dropna():
            all_genres.extend(g)
        genre_counts = pd.Series(all_genres).value_counts().head(15)
        fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                     title='Топ-15 жанров')
        st.plotly_chart(fig, use_container_width=True)

# ========== СТРАНИЦА 3: РЕКОМЕНДАЦИИ ==========
elif page == "🎯 Рекомендации":
    st.header("Система рекомендаций")

    rec_type = st.radio("Тип рекомендаций",
                        ["🎬 Похожие фильмы", "🔥 Популярные", "✨ Персонализированные"])

    if rec_type == "🎬 Похожие фильмы":
        movie_titles = movies['title'].dropna().sort_values().tolist()
        selected = st.selectbox("Выберите фильм", movie_titles)
        movie_id = movies[movies['title'] == selected]['id'].values[0]

        if st.button("Найти похожие"):
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

    elif rec_type == "🔥 Популярные":
        col1, col2 = st.columns(2)
        with col1:
            genres_list = []
            if 'genres_parsed' in movies.columns:
                for g in movies['genres_parsed'].dropna():
                    genres_list.extend(g)
            genres_list = sorted(set(genres_list))
            genre = st.selectbox("Жанр (опционально)", ["Все"] + genres_list)
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
        selected_genres = st.multiselect("Любимые жанры", genres_list, default=["Drama", "Action"])
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

# ========== СТРАНИЦА 4: ТРЕНДЫ ==========
else:
    st.header("Тренды киноиндустрии")
    if 'release_year' in movies_fe.columns:
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