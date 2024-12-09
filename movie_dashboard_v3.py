import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
import sqlite3
import os

# ==================================
# Load Dataset Files into SQL Database
# ==================================
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

load_data_to_sql()


def view_exists(conn, view_name):
    query = f"""
        SELECT name FROM sqlite_master WHERE type='view' AND name='{view_name}';
    """
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchone() is not None


# ==================================
# Precompute Recommendations
# ==================================
class NCF(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        pred = nn.Sigmoid()(self.output(vector))
        return pred

@st.cache_resource
def load_model(file_path, num_users, num_items):
    model = NCF(num_users, num_items)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

def precompute_recommendations():
    conn = sqlite3.connect("movie_recommendation.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recommendations';")
    if cursor.fetchone() is None:
        print("Precomputing recommendations...")
        ratings = pd.read_sql("SELECT * FROM ratings", conn)
        num_users = ratings['UserID'].max() + 1
        num_items = ratings['MovieID'].max() + 1
        model = load_model("mrs-v4.pkl", num_users, num_items)

        recommendations = []
        for user_id in range(1, num_users):
            interacted_items = ratings[ratings["UserID"] == user_id]["MovieID"].tolist()
            not_interacted_items = set(range(1, num_items)) - set(interacted_items)
            test_items = list(np.random.choice(list(not_interacted_items), 99))
            if interacted_items:
                test_items.append(interacted_items[0])

            user_tensor = torch.tensor([user_id] * 100)
            item_tensor = torch.tensor(test_items)
            predicted_labels = model(user_tensor, item_tensor).detach().numpy().squeeze()
            top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][:10]]

            for rank, item in enumerate(top10_items, start=1):
                recommendations.append((user_id, item, rank))

        recommendations_df = pd.DataFrame(recommendations, columns=["UserID", "MovieID", "Rank"])
        recommendations_df.to_sql("recommendations", conn, if_exists="replace", index=False)
    else:
        print("Recommendations already exist in the database. Skipping precomputation.")
    conn.close()

precompute_recommendations()

# ==================================
# Streamlit Dashboard
# ==================================
# Connect to SQLite database
conn = sqlite3.connect("movie_recommendation.db")

# Dropdown menu for sections
section = st.sidebar.selectbox(
    "Choose a section:",
    ["MovieLens Analysis", "Recommendation System"]
)
if section == "MovieLens Analysis":
    # =========================
    # Data Analysis Section
    # =========================
    st.title("MovieLens Dataset Analysis")

    # Helper functions for analysis
    def preprocess_movies():
        movies = pd.read_sql("SELECT * FROM movies", conn)
        movies['Genres'] = movies['Genres'].str.split('|')
        return movies.explode('Genres')

    def get_top_rated_movies():
        ratings = pd.read_sql("SELECT * FROM ratings", conn)
        movies = pd.read_sql("SELECT * FROM movies", conn)
        avg_ratings = ratings.groupby("MovieID")["Rating"].mean().sort_values(ascending=False).head(10)
        top_rated = movies[movies["MovieID"].isin(avg_ratings.index)]
        return pd.merge(top_rated, avg_ratings, on="MovieID")

    def get_most_reviewed_movies():
        ratings = pd.read_sql("SELECT * FROM ratings", conn)
        movies = pd.read_sql("SELECT * FROM movies", conn)
        review_counts = ratings.groupby("MovieID").size().sort_values(ascending=False).head(10)
        review_counts = review_counts.rename("Review Count")  # Assign a name to the Series
        most_reviewed = movies[movies["MovieID"].isin(review_counts.index)]
        return pd.merge(most_reviewed, review_counts, left_on="MovieID", right_index=True)

    def get_ratings_distribution():
        ratings = pd.read_sql("SELECT * FROM ratings", conn)
        return ratings["Rating"].value_counts().sort_index()

    def get_average_rating_vs_reviews():
        ratings = pd.read_sql("SELECT * FROM ratings", conn)
        movies = pd.read_sql("SELECT * FROM movies", conn)
        avg_ratings = ratings.groupby("MovieID")["Rating"].mean()
        review_counts = ratings.groupby("MovieID").size()
        return pd.DataFrame({"Average Rating": avg_ratings, "Review Count": review_counts}).reset_index()

    def get_ratings_by_genre():
        ratings = pd.read_sql("SELECT * FROM ratings", conn)
        genre_movies = preprocess_movies()
        merged = pd.merge(genre_movies, ratings, on="MovieID")
        grouped = merged.groupby("Genres")["Rating"].mean().reset_index()
        grouped.columns = ["Genres", "Average Rating"]
        return grouped

    # Ratings Distribution Section
    st.subheader("Ratings Distribution")
    ratings_dist = get_ratings_distribution()
    st.bar_chart(ratings_dist)

    # Most Reviewed Movies Section
    st.subheader("Top 10 Most Reviewed Movies")
    most_reviewed = get_most_reviewed_movies()
    st.table(most_reviewed)

    # Average Rating vs. Review Count Section
    st.subheader("Average Rating vs. Review Count")
    avg_rating_vs_reviews = get_average_rating_vs_reviews()
    st.scatter_chart(avg_rating_vs_reviews, x="Review Count", y="Average Rating")
    # Ratings by Genre Section
    st.subheader("Ratings by Genre")
    ratings_by_genre = get_ratings_by_genre()
    st.bar_chart(ratings_by_genre.set_index("Genres"))

    # Top 10 Rated Movies Section
    st.subheader("Top 10 Rated Movies")
    top_rated_movies = get_top_rated_movies()
    st.table(top_rated_movies)

if section == "Recommendation System":
    # =========================
    # Recommendation System Section
    # =========================
    st.title("Movie Recommendation System")

    # Input User ID
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)

    # Dropdown to choose between recommendations or top-rated movies
    action = st.selectbox(
        "Choose an action:",
        ["Get Top 10 Recommendations", "Get Top 10 Rated Movies"]
    )

    if action == "Get Top 10 Recommendations":
        if st.button("Fetch Recommendations"):
            query = f"""
                SELECT movies.Title, movies.Genres
                FROM recommendations
                JOIN movies ON recommendations.MovieID = movies.MovieID
                WHERE recommendations.UserID = {user_id}
                ORDER BY recommendations.Rank
                LIMIT 10
            """
            recommendations = pd.read_sql(query, conn)

            if recommendations.empty:
                st.write("No recommendations available for this user.")
            else:
                st.subheader(f"Top 10 Recommendations for User {user_id}")
                for idx, row in recommendations.iterrows():
                    st.write(f"{idx+1}. **{row['Title']}** (Genre: {row['Genres']})")

    elif action == "Get Top 10 Rated Movies":
        if st.button("Fetch Top Rated Movies"):
            query = f"""
                SELECT movies.Title, movies.Genres, ratings.Rating
                FROM ratings
                JOIN movies ON ratings.MovieID = movies.MovieID
                WHERE ratings.UserID = {user_id}
                ORDER BY ratings.Rating DESC, ratings.Timestamp DESC
                LIMIT 10
            """
            top_rated_movies = pd.read_sql(query, conn)

            if top_rated_movies.empty:
                st.write("No data available for this user.")
            else:
                st.subheader(f"Top 10 Rated Movies for User {user_id}")
                for idx, row in top_rated_movies.iterrows():
                    st.write(f"{idx+1}. **{row['Title']}** - Rating: {row['Rating']} (Genre: {row['Genres']})")

