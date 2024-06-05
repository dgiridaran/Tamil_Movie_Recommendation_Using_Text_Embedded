# Tamil_Movie_Recommendation_Based_On_Similar_Plots

[demo_video.webm](https://github.com/dgiridaran/Tamil_Movie_Recommendation_Using_Text_Embedded/assets/58911376/cd963776-2c19-4355-a2b8-2ecc0e66f6ce)

This is a movie recommender system for Tamil movies, built using SBERT embedding for recommendation based on similar plots, and TFIDF embedding for searching by title of the movie. 

# Data Collection
Data was collected by scrapping Tamil movies from 1990 to 2024 from the IMDB website.

# Preprocessing
- I cleaned the text data by removing the nonalphabetic or non-numeric values and lowercase corpus.
- Encoded the movie Title using TFIDF and stored it in the Numpy matrix.
- Encoded the movie plots using the SBERT Model and stored the encodings in the Numpy matrix.
- Built a KNN model using this Encoded movie plots.

# Working
- The Search works based on cosine similarity, First, we encode the incoming search by TFIDF encoder do cosine similarity with previously encoded data and get the index of the highest values.
- Using that index value we can get the movie data from the data frame.
- From this data we can get the plot, Then encode the plot using SBERT and using previously trained KNN models we can get the vectors nearest to our input vector (embedding), In our case we took to 5 nearest vector and their indexes.
- using that index value we can get the movie data from the dataframe.

**You can use the app by [clicking here](https://tamil-movie-recommendation-using-text-embedded-giridaran.streamlit.app/)**

