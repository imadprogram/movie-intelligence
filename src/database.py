from pymongo import MongoClient
import json


client = MongoClient('mongodb://localhost:27017')

db = client["movie_db"]

collection = db["movies"]
def insert_movies():

    with open('data/processed/movies_cleaned.json' , 'r') as f:

        movies = json.load(f)


    collection.drop()
    result = collection.insert_many(movies)

    print(len(result.inserted_ids))

insert_movies()