# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:55:19 2021

@author: junaid
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras, tensordot, nn

#impor data
movie = pd.read_csv("movies.csv")
rating = pd.read_csv("ratings.csv")

#Cleaning

#Removing the years from the 'title' column
movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '')

#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movie['title'] = movie['title'].apply(lambda x: x.strip())

#merging both data frames
combined_data = pd.merge(movie, rating).drop(['genres', 'timestamp'], axis = 1)

#Encode UserId
user_ids = combined_data["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

#Encode MovieId
movie_ids = combined_data["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

#Replacing both encoded values
combined_data["user"] = combined_data["userId"].map(user2user_encoded)
combined_data["movie"] = combined_data["movieId"].map(movie2movie_encoded)

#setting parameters for model
num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)

#Convert all ratings to float type
combined_data["rating"] = combined_data["rating"].values.astype(np.float32)
#Save this dataframe to pkl
combined_data.to_pickle('dataset')

# min and max ratings will be used to normalize the ratings later
min_rating = min(combined_data["rating"])
max_rating = max(combined_data["rating"])

#Extracting user and movie data for training
combined_data = combined_data.sample(frac=1, random_state=42)
x = combined_data[["user", "movie"]].values

# Normalize the targets between 0 and 1. Makes it easy to train.
y = combined_data["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.9 * combined_data.shape[0])

#Split data into training and validation
x_train, x_val, y_train, y_val = (x[:train_indices],x[train_indices:],
                                  y[:train_indices],y[train_indices:])

#Set embedding size
EMBEDDING_SIZE = 50

#class for training the recommender
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        
        super(RecommenderNet, self).__init__(**kwargs)
        
        self.num_users = num_users
        self.num_movies = num_movies
        
        self.embedding_size = embedding_size
        
        self.user_embedding = keras.layers.Embedding(num_users, embedding_size, embeddings_initializer=keras.initializers.glorot_normal, 
                                                     embeddings_regularizer=keras.regularizers.l2(1e-6))
        
        self.user_bias = keras.layers.Embedding(num_users, 1)
        
        self.movie_embedding = keras.layers.Embedding(num_movies, embedding_size, embeddings_initializer="he_normal",
                                                      embeddings_regularizer=keras.regularizers.l2(1e-6))
        
        self.movie_bias = keras.layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        #compute tensor_dotproduct of user and movie vector
        dot_user_movie = tensordot(user_vector, movie_vector, 2)
        
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        
        # The sigmoid activation forces the rating to between 0 and 1
        return nn.sigmoid(x)

#Model creation
model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)

#Compile Model
model.compile(loss = keras.losses.BinaryCrossentropy(), 
              optimizer=keras.optimizers.Adam(lr=0.01, beta_2 = 0.001, epsilon = 1e-5, amsgrad = True))

#Start Training
history = model.fit(x=x_train, y=y_train, batch_size=512,
                    epochs=10, verbose=1, validation_data=(x_val, y_val))

#Save Training History
np.save('training_hist.npy', history.history)

#Saving Model Weights
model.save_weights('final_trained_weights')


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

