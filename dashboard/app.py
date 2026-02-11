import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import TMDBDataLoader
from src.features.engineering import FeatureEngineer
from src.models.recommender import HybridMovieRecommender
import pickle
import os

# Конфигурация страницы
st.set_page_config(
    page_title="TMDB Movie Analytics",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_data_and_model():
    loader = TMDBDataLoader()
    movies_df = loader.load_data()

    engineer = FeatureEngineer()
    movies_with_features = engineer.create_features(movies_df)

    recommender = HybridMovieRecommender(n_recommendations=10)
    recommender.fit(movies_with_features)

    return movies_df, movies_with_features, recommender


def main():
    st.title("🎬 TMDB Movie Analytics & Recommendation System")
    st.markdown("---")

    with st.spinner("Загрузка данных и обучение модели..."):
        movies_df, movies_with_features, recommender = load_data_and_model()

    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Выберите раздел:",
        ["📊 Обзор данных", "🔍 Аналитика", "🎯 Рекомендации", "📈 Тренды"]
    )

    if page == "📊 Обзор данных":
        st.header("Обзор датасета TMDB Movies")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего фильмов", len(movies_df))
        with col2:
            st.metric("Средний рейтинг", f"{movies_df['vote_average'].mean():.2f}")
        with col3:
            st.metric("Общий бюджет", f"${movies_df['budget'].sum() / 1e9:.1f}B")

        st.subheader("Просмотр данных")
        if st.checkbox("Показать сырые данные"):
            st.dataframe(movies_df.head(100))

        st.subheader("Распределение рейтингов")
        fig = px.histogram(movies_df, x='vote_average', nbins=50,
                           title="Distribution of Movie Ratings",
                           labels={'vote_average': 'Rating'})
        st.plotly_chart(fig, use_container_width=True)

    elif page == "🔍 Аналитика":
        st.header("Аналитика фильмов")

        # Выбор метрик для анализа
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Ось X:",
                                  ['budget', 'revenue', 'vote_average', 'popularity', 'release_year'])
        with col2:
            y_axis = st.selectbox("Ось Y:",
                                  ['revenue', 'vote_average', 'popularity', 'vote_count'])

        fig = px.scatter(movies_with_features,
                         x=x_axis,
                         y=y_axis,
                         color='vote_average',
                         size='vote_count',
                         hover_data=['title', 'genres'],
                         title=f"{y_axis} vs {x_axis}",
                         log_x=True if x_axis in ['budget', 'revenue'] else False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Анализ по жанрам")
        # Извлекаем все жанры
        all_genres = []
        for genres in movies_df['genres']:
            all_genres.extend(genres)
        genre_counts = pd.Series(all_genres).value_counts()

        fig = px.bar(x=genre_counts.index[:15], y=genre_counts.values[:15],
                     title="Топ 15 жанров")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "🎯 Рекомендации":
        st.header("Система рекомендаций фильмов")

        rec_type = st.radio(
            "Тип рекомендаций:",
            ["🎬 Похожие фильмы", "🔥 Популярные", "✨ Персонализированные"]
        )

        if rec_type == "🎬 Похожие фильмы":
            st.subheader("Найдите похожие фильмы")

            # Поиск фильма
            movie_title = st.selectbox(
                "Выберите фильм:",
                movies_df['title'].sort_values().tolist()
            )

            if movie_title:
                movie_id = movies_df[movies_df['title'] == movie_title]['id'].values[0]

                if st.button("Найти похожие фильмы"):
                    recommendations = recommender.recommend_by_content(movie_id)

                    st.subheader(f"Фильмы, похожие на '{movie_title}':")

                    # Отображаем рекомендации в виде карточек
                    cols = st.columns(3)
                    for idx, (_, row) in enumerate(recommendations.iterrows()):
                        with cols[idx % 3]:
                            st.markdown(f"""
                            <div style="padding: 10px; border-radius: 10px; border: 1px solid #ddd; margin: 5px;">
                                <h4>{row['title']}</h4>
                                <p>📅 {row['release_year']}</p>
                                <p>⭐ {row['vote_average']}/10</p>
                                <p>🎭 {', '.join(row['genres'][:2])}</p>
                                <p>🔗 Сходство: {row['similarity_score']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)

        elif rec_type == "🔥 Популярные":
            st.subheader("Популярные фильмы")

            col1, col2 = st.columns(2)
            with col1:
                genre = st.selectbox("Жанр:",
                                     ["Все"] + sorted(list(set(all_genres))))
            with col2:
                year_range = st.slider("Годы выпуска:",
                                       min_value=1910,
                                       max_value=2020,
                                       value=(2000, 2020))

            genre_filter = None if genre == "Все" else genre
            recommendations = recommender.recommend_by_popularity(
                genre=genre_filter,
                year_range=year_range
            )

            st.dataframe(recommendations)

        else:
            st.subheader("Персонализированные рекомендации")

            st.write("Укажите ваши предпочтения:")

            selected_genres = st.multiselect(
                "Любимые жанры:",
                options=sorted(list(set(all_genres))),
                default=["Action", "Drama"]
            )

            min_rating = st.slider("Минимальный рейтинг:", 0.0, 10.0, 7.0, 0.5)
            min_year = st.number_input("Минимальный год выпуска:", 1900, 2020, 2000)

            if st.button("Получить рекомендации"):
                preferences = {
                    'genres': selected_genres,
                    'min_rating': min_rating,
                    'year_range': (min_year, 2020)
                }

                recommendations = recommender.recommend_by_preferences(preferences)

                st.success(f"Найдено {len(recommendations)} рекомендаций")

                fig = px.bar(recommendations.head(10),
                             x='title',
                             y='vote_average',
                             color='vote_count',
                             title="Топ 10 рекомендаций",
                             labels={'vote_average': 'Рейтинг', 'title': 'Фильм'})
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.header("Тренды в киноиндустрии")

        yearly_stats = movies_with_features.groupby('release_year').agg({
            'revenue': 'sum',
            'budget': 'sum',
            'vote_average': 'mean',
            'id': 'count'
        }).rename(columns={'id': 'movie_count'}).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly_stats['release_year'],
                                 y=yearly_stats['revenue'],
                                 mode='lines+markers',
                                 name='Выручка',
                                 yaxis='y'))
        fig.add_trace(go.Bar(x=yearly_stats['release_year'],
                             y=yearly_stats['movie_count'],
                             name='Количество фильмов',
                             yaxis='y2',
                             opacity=0.3))

        fig.update_layout(
            title="Тренды киноиндустрии по годам",
            xaxis_title="Год",
            yaxis_title="Выручка ($)",
            yaxis2=dict(title="Количество фильмов",
                        overlaying='y',
                        side='right'),
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()