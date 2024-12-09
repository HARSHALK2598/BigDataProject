import pandas as pd
import sqlite3
import os

def load_data_to_sql():
    if not os.path.exists("movie_recommendation.db"):
        # Create database and load data only if it doesn't exist
        conn = sqlite3.connect("movie_recommendation.db")
        ratings = pd.read_csv('ml-1m/ratings.dat', sep="::", names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python', encoding='ISO-8859-1')
        movies = pd.read_csv('ml-1m/movies.dat', sep="::", names=["MovieID", "Title", "Genres"], engine='python', encoding='ISO-8859-1')
        ratings.to_sql("ratings", conn, if_exists="replace", index=False)
        movies.to_sql("movies", conn, if_exists="replace", index=False)
        conn.close()
        print("Database created and data loaded.")
    else:
        print("Database already exists. Skipping data load.")

if __name__ == "__main__":
    load_data_to_sql()

