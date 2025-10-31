import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app) 

print("Loading data and building model...")

try:
    movies_df = pd.read_csv("movies.csv")
except Exception as e:
    print(f"Error loading movies.csv: {e}")
    movies_df = pd.DataFrame(columns=['id', 'title', 'genres']) 

movies_df['genres'] = movies_df['genres'].fillna('')

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'))
genre_matrix = vectorizer.fit_transform(movies_df['genres'])


cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

indices = pd.Series(movies_df.index, index=movies_df['id']).astype(int)

print("Model built successfully.")

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # Get the list of movie IDs from the request body
    data = request.get_json()
    if not data or 'watchlist_ids' not in data:
        return jsonify({'error': 'Missing watchlist_ids in request body'}), 400
    
    watchlist_ids = data['watchlist_ids']
    
   
    try:
        
        watchlist_indices = [indices[int(movie_id)] for movie_id in watchlist_ids if int(movie_id) in indices]
        
    except Exception as e:
        print(f"Error finding indices: {e}")
        return jsonify({'error': f'Error finding indices: {e}'}), 500

    if not watchlist_indices:
        print(f"Watchlist IDs {watchlist_ids} had no matches in the local movie data.")
        return jsonify({'recommendations': []})

  
    sim_scores = cosine_sim[watchlist_indices].mean(axis=0)

  
    sim_scores = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:21] 

    movie_indices = [i[0] for i in sim_scores]

    recommended_movie_ids = movies_df['id'].iloc[movie_indices].tolist()
    
   
    watchlist_ids_int = [int(id) for id in watchlist_ids]
    final_recommendations = [id for id in recommended_movie_ids if id not in watchlist_ids_int]
    
   
    return jsonify({'recommendations': final_recommendations})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
