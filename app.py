import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('data/tamil_movies_1990_2024_cleaned.csv')
st.title('Movie Recommender System')
movie_name = st.text_input("Enter the movie name: ")
search_butten = st.button("Search")

def search_by_name(movie_name):
    title = movie_name.lower()
    with open('models/vectorize_search.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('models/tfidf_values.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    search_index = similarity.argsort()[-1:][::-1]
    return df.iloc[search_index]

def recommend_movies(context):
    # Find k nearest neighbors
    with open('models/knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embadded_context = model.encode(context)
    distances, indices = knn_model.kneighbors([embadded_context])
    # Return the indices of the nearest neighbors (movies)
    # print(indices)
    idx = indices[0][1:]
    return df.iloc[idx]

if movie_name and search_butten:
    st.title("Search Results: ")
    search_result = search_by_name(movie_name)
    img_url = search_result['Image'].values[0]
    if img_url is np.nan:
        img_url = 'images/no_image.png'
    movie_title = search_result['Title_new'].values[0]
    title_url = search_result['Title_URL'].values[0]
    release_year = search_result['Title1'].values[0]
    rating = search_result['ipcratingstar'].values[0]
    content = search_result['Content'].values[0]
    st.image(img_url, width=300)
    st.write("**Movie name:**",movie_title.capitalize())
    st.write("**Releace Year:**",release_year)
    st.write("**Rating:**",rating)
    st.write("**content:**",content)
    st.markdown("[show more](%s)" % title_url)
    st.title("Recommended Movies: ")
    content = search_result['Clean_content'].values[0]
    recomended_movies = recommend_movies(content)
    for i in range(5):
        if recomended_movies['Image'].iloc[i] is not np.nan:
            # print(recomended_movies['Image'].iloc[i])

            st.image(recomended_movies['Image'].iloc[i], width=300)
        else:
            st.image('images/no_image.png', width=300)
        st.write("**Movie name:**",recomended_movies['Title_new'].iloc[i])
        st.write("**Releace Year:**",recomended_movies['Title1'].iloc[i])
        st.write("**Rating:**",recomended_movies['ipcratingstar'].iloc[i])
        st.write("**content:**",recomended_movies['Content'].iloc[i])
        st.markdown("[show more](%s)" % recomended_movies['Title_URL'].iloc[i])
        st.markdown("---")


