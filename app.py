import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel 

app = Flask(__name__)
CORS(app) 

print("Loading data and building model...")

# Load the movie dataset
try:
    movies_df = pd.read_csv("movies.csv")
except Exception as e:
    print(f"Error loading movies.csv: {e}")
    movies_df = pd.DataFrame(columns=['id', 'title', 'genres']) 

movies_df['genres'] = movies_df['genres'].fillna('')

# --- Printing all unique genres and counts in terminal ---
if not movies_df.empty:
    all_genres = movies_df['genres'].str.split(';').explode().str.strip()
    genre_counts = all_genres.value_counts()

    print("\n All Genres Found in Dataset:")
    print("-----------------------------------")
    for genre, count in genre_counts.items():
        print(f"• {genre} ({count} movies)")
    print("-----------------------------------")
    print(f"Total unique genres: {genre_counts.count()}\n")
else:
    print(" No movie data found to analyze genres.\n")

# Building the vectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'))
genre_matrix = vectorizer.fit_transform(movies_df['genres'])

indices = pd.Series(movies_df.index, index=movies_df['id']).astype(int)

print("Model built successfully.\n")

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    if not data or 'watchlist_ids' not in data:
        return jsonify({'error': 'Missing watchlist_ids in request body'}), 400
    
    watchlist_ids = data['watchlist_ids']
    
    #   indices for the user's watchlist movies 
    try:
        watchlist_indices = [indices[int(movie_id)] for movie_id in watchlist_ids if int(movie_id) in indices]
    except Exception as e:
        print(f"Error finding indices: {e}")
        return jsonify({'error': f'Error finding indices: {e}'}), 500

    if not watchlist_indices:
        print(f"Watchlist IDs {watchlist_ids} had no matches in the local movie data.")
        return jsonify({'recommendations': []})

    # Get genre vectors for watchlist movies
    watchlist_vectors = genre_matrix[watchlist_indices]

    user_profile_vector = watchlist_vectors.mean(axis=0).A 

    sim_scores = linear_kernel(user_profile_vector, genre_matrix)[0]

    sim_scores = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)[1:21]
    movie_indices = [i[0] for i in sim_scores]

    # Get recommended IDs
    recommended_movie_ids = movies_df['id'].iloc[movie_indices].tolist()

    watchlist_ids_int = [int(id) for id in watchlist_ids]
    final_recommendations = [id for id in recommended_movie_ids if id not in watchlist_ids_int]
    
    recommendations_with_names = [
        {
            'id': int(row['id']),
            'title': row['title']
        }
        for _, row in movies_df[movies_df['id'].isin(final_recommendations)].iterrows()
    ]

    # Return both IDs and names
    return jsonify({'recommendations': recommendations_with_names})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "StreamHub ML Service (Genre-Based) is running"})


if __name__ == '__main__':
    app.run(port=5001)
