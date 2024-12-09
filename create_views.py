def create_views(conn):
    cursor = conn.cursor()

    # View for top-rated movies
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS top_rated_movies AS
        SELECT movies.MovieID, movies.Title, AVG(ratings.Rating) AS AverageRating, movies.Genres
        FROM ratings
        JOIN movies ON ratings.MovieID = movies.MovieID
        GROUP BY movies.MovieID
        ORDER BY AverageRating DESC
        LIMIT 10;
    """)

    # View for most reviewed movies
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS most_reviewed_movies AS
        SELECT movies.MovieID, movies.Title, COUNT(ratings.Rating) AS ReviewCount, movies.Genres
        FROM ratings
        JOIN movies ON ratings.MovieID = movies.MovieID
        GROUP BY movies.MovieID
        ORDER BY ReviewCount DESC
        LIMIT 10;
    """)

    # View for ratings distribution
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS ratings_distribution AS
        SELECT ratings.Rating, COUNT(*) AS Count
        FROM ratings
        GROUP BY ratings.Rating;
    """)

    # View for average rating vs. review count
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS avg_rating_vs_review_count AS
        SELECT movies.MovieID, movies.Title, AVG(ratings.Rating) AS AverageRating, COUNT(ratings.Rating) AS ReviewCount
        FROM ratings
        JOIN movies ON ratings.MovieID = movies.MovieID
        GROUP BY movies.MovieID;
    """)

    # Create a separate table for exploded genres if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exploded_genres AS
        SELECT movies.MovieID, movies.Title, TRIM(genre) AS Genre
        FROM movies, 
        (SELECT TRIM(SUBSTR(movies.Genres, INSTR(movies.Genres || '|', '|') - LENGTH(movies.Genres) + 1)) AS genre
         FROM movies) genres
        WHERE genres.genre IS NOT NULL;
    """)

    # View for ratings by genre
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS ratings_by_genre AS
        SELECT g.Genre, AVG(r.Rating) AS AverageRating
        FROM exploded_genres g
        JOIN ratings r ON g.MovieID = r.MovieID
        GROUP BY g.Genre;
    """)

    conn.commit()
    print("Views created successfully.")

