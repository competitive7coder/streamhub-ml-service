import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
# --- CHANGE: Use linear_kernel for memory efficiency ---
from sklearn.metrics.pairwise import linear_kernel 

# Initialize the Flask app
app = Flask(__name__)
CORS(app) 

# --- 1. Load Data and Build ML Model (This runs once at startup) ---
print("Loading data and building model...")

# Load the movie dataset
try:
    movies_df = pd.read_csv("movies.csv")
except Exception as e:
    print(f"Error loading movies.csv: {e}")
    movies_df = pd.DataFrame(columns=['id', 'title', 'genres']) # Empty dataframe on error

movies_df['genres'] = movies_df['genres'].fillna('')

# The Vectorizer is kept, as it's small and necessary
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'))
# genre_matrix is a Sparse Matrix, which is memory-efficient
genre_matrix = vectorizer.fit_transform(movies_df['genres'])

# --- IMPORTANT: The full cosine_sim matrix is NOT calculated here. ---

# Create a helper series to map movie IDs to their index in the dataframe
indices = pd.Series(movies_df.index, index=movies_df['id']).astype(int)

print("Model built successfully.")

# --- 2. Create the Recommendation API Endpoint ---
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    if not data or 'watchlist_ids' not in data:
        return jsonify({'error': 'Missing watchlist_ids in request body'}), 400
    
    watchlist_ids = data['watchlist_ids']
    
    # --- 3. The ML Logic: Build User's "Taste Profile" (REVISED FOR MEMORY) ---
    
    try:
        # Get the dataframe indices for the movies in the user's watchlist
        watchlist_indices = [indices[int(movie_id)] for movie_id in watchlist_ids if int(movie_id) in indices]
        
    except Exception as e:
        print(f"Error finding indices: {e}")
        return jsonify({'error': f'Error finding indices: {e}'}), 500

    if not watchlist_indices:
        print(f"Watchlist IDs {watchlist_ids} had no matches in the local movie data.")
        return jsonify({'recommendations': []})

    # Get the *sparse* genre vectors for all movies in the watchlist
    watchlist_vectors = genre_matrix[watchlist_indices]

    # Calculate the mean (average) vector. This is the user's taste profile.
    # We explicitly convert the result to a dense array for the next step
    user_profile_vector = watchlist_vectors.mean(axis=0).A 

    # Calculate Similarity between the single user_profile_vector and ALL movie vectors.
    # linear_kernel is ideal for this sparse-to-dense calculation.
    sim_scores = linear_kernel(user_profile_vector, genre_matrix)
    
    # The result is a 1xN array, so we take the first row
    sim_scores = sim_scores[0] 

    # Sort the movies based on the similarity scores
    sim_scores = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)

    # Get the scores of the 20 most similar movies (skip the first one, which is the profile itself)
    sim_scores = sim_scores[1:21] 

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
    return jsonify({"message": "✅ StreamHub ML Service (Genre-Based) is running"})

# --- 6. Run the App ---
if __name__ == '__main__':
    # This part is for local development
    app.run(port=5001)
