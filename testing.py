# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:32:07 2021

@author: junaid
"""
import pandas as pd
import numpy as np
from training import RecommenderNet

movie_df = pd.read_csv("movies.csv")
combined_data = pd.read_pickle('dataset')

#Encode UserId
user_ids = combined_data["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

#Encode MovieId
movie_ids = combined_data["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

#setting parameters for model
num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
EMBEDDING_SIZE = 50

#Loading model and saved weights
model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model.load_weights('final_trained_weights')

# Let us get a user and see the top recommendations.
user_id = combined_data.userId.sample(1).iloc[0]
movies_watched_by_user = combined_data[combined_data.userId == user_id]
movies_not_watched = movie_df[
    ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
]["movieId"]
movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

user_encoder = user2user_encoded.get(user_id)

user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))

ratings = model.predict(user_movie_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]

recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]

print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Movies with high ratings from user")
print("----" * 8)
top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ":", row.genres)

print("----" * 8)
print("Top 10 movie recommendations")
print("----" * 8)
recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)
