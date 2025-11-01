import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os # Added for port binding

# Initialize the Flask app
app = Flask(__name__)
# Allow cross-origin requests (from your Node.js server)
CORS(app) 

# --- 1. Load Data and Build ML Model (This runs once at startup) ---
print("Loading data and building model...")

# Load the movie dataset
try:
    movies_df = pd.read_csv("movies.csv")
except Exception as e:
    print(f"Error loading movies.csv: {e}")
    movies_df = pd.DataFrame(columns=['id', 'title', 'genres']) # Empty dataframe on error

# Replace empty genres with a placeholder
movies_df['genres'] = movies_df['genres'].fillna('')

# Create a "bag of words" for the genres (e.g., "Drama;Crime" -> ["Drama", "Crime"])
# This is the model that matches your movies.csv
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'))
genre_matrix = vectorizer.fit_transform(movies_df['genres'])

# Calculate the similarity between all movies based on their genres
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Create a helper series to map movie IDs to their index in the dataframe
# This is crucial for fast lookups
indices = pd.Series(movies_df.index, index=movies_df['id']).astype(int)

print("Model built successfully.")

# --- 2. Create the Recommendation API Endpoint ---
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # Get the list of movie IDs from the request body
    data = request.get_json()
    if not data or 'watchlist_ids' not in data:
        return jsonify({'error': 'Missing watchlist_ids in request body'}), 400
    
    watchlist_ids = data['watchlist_ids']
    
    # --- 3. The ML Logic: Build User's "Taste Profile" ---
    
    try:
        # We check if the movie_id exists in our 'indices' map before trying to use it.
        watchlist_indices = [indices[int(movie_id)] for movie_id in watchlist_ids if int(movie_id) in indices]
        
    except Exception as e:
        print(f"Error finding indices: {e}")
        return jsonify({'error': f'Error finding indices: {e}'}), 500

    if not watchlist_indices:
        # If watchlist is empty or has movies not in our dataset, return an empty list
        print(f"Watchlist IDs {watchlist_ids} had no matches in the local movie data.")
        return jsonify({'recommendations': []})

    # Get the genre similarity scores for all movies in the watchlist
    sim_scores = cosine_sim[watchlist_indices].mean(axis=0)

    # Sort the movies based on the similarity scores
    sim_scores = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)

    # Get the scores of the 20 most similar movies
    sim_scores = sim_scores[1:21] # Get top 20 (skip the first one, which is 1.0)

    # Get the movie indices from the scores
    movie_indices = [i[0] for i in sim_scores]

    # --- 4. Filter and Return ---
    
    # Get the IDs of the recommended movies
    recommended_movie_ids = movies_df['id'].iloc[movie_indices].tolist()
    
    # Filter out movies that are *already* in the user's watchlist
    watchlist_ids_int = [int(id) for id in watchlist_ids]
    final_recommendations = [id for id in recommended_movie_ids if id not in watchlist_ids_int]
    
    # Return the final list of movie IDs
    return jsonify({'recommendations': final_recommendations})

# --- 5. Add a Home Route ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "StreamHub ML Service (Genre-Based) is running"})

# --- 6. Run the App ---
if __name__ == '__main__':
    # This part is for local development, Gunicorn runs the 'app' variable directly on Render
    # Get the port from the environment, defaulting to 5001 for local
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
