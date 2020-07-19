import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Functions: 
def get_title_from_index(index):
      return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
      return df[df.title == title]["index"].values[0]


# Read CSV File
df=pd.read_csv("movie_dataset.csv") 
# print(df.head(n=10))  print(df.columns) 

#  Select Features
features=['keywords','cast','genres','director']

# Create a column in DF which combines all selected features

for feature in features:
   df[feature]=df[feature].fillna('')
def combined_features(row):
      try:
         return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
      except:
         print("Error: ",row)
         
df["combined_freatures"]=df.apply(combined_features,axis=1)
# print(df["combined_freatures"].head())

# Create count matrix from this new combined column
cv=CountVectorizer() 
count_matrix=cv.fit_transform(df["combined_freatures"])
 
# Compute the Cosine Similarity based on the count_matrix
cosine_sim=cosine_similarity(count_matrix)
movie_user_likes = input("Enter the movie name: ")

# Get index of this movie from its title
movie_index=get_index_from_title(movie_user_likes)
similar_movies=list(enumerate(cosine_sim[movie_index]))

# Get a list of similar movies in descending order of similarity score
sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

# Print titles of first 20 movies
i=0
for movie in sorted_similar_movies:
   if(i==0):
         pass
   else:
         print(get_title_from_index(movie[0]))
   i=i+1
   if(i>20):
      break
   
