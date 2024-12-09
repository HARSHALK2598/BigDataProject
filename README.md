# **Big Data Project: Movie Recommendation System**

This project implements a movie recommendation system using pre-trained models and a user-friendly Streamlit dashboard. The system utilizes SQL databases to store movie data, user ratings, and precomputed recommendations for efficient querying.

---

## **Project Structure**

- **`ml-1m/`**:
  - Directory containing the MovieLens 1M dataset.
  - Refer to the `README` file in this folder for dataset details.
- **`BigData_Project_v4.ipynb`**:
  - Jupyter notebook for training the Neural Collaborative Filtering (NCF) model.
  - The trained model weights are saved as `mrs-v4.pkl`.
- **`movie_recommendation.db`**:
  - SQLite database storing the `movies`, `ratings`, and `recommendations` tables.
- **`movie_dashboard_v3.py`**:
  - Streamlit dashboard script for interacting with the recommendation system.
- **`load_data_to_sql.py`**:
  - Script for loading the dataset into the SQLite database.
- **`create_views.py`**:
  - Script for creating reusable SQL views to speed up dashboard queries.
- **`precompute_recommendations.py`**:
  - Script to precompute and store the top 10 movie recommendations for all users.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/BigDataProject.git
cd BigDataProject
