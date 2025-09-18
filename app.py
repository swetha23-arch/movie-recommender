import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies_data = load_data()

# Preprocess
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + " " + movies_data['keywords'] + " " + movies_data['tagline'] + " " + movies_data['cast'] + " " + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# Recommendation function
def recommend_movies(movie_name, num_recommendations=10):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return ["No close match found. Try another movie."]

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:num_recommendations+1]):  # skip the first one (same movie)
        index = movie[0]
        recommended_movies.append(movies_data.iloc[index].title)

    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Get top movie suggestions based on your favorite movie.")

movie_name = st.text_input("Enter a movie you like:")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations = recommend_movies(movie_name)
        st.subheader("Movies suggested for you:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

