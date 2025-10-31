from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)
CORS(app)

# --- Load movies data ---
try:
    movies_df = pd.read_csv("movies.csv")
except FileNotFoundError:
    movies_df = pd.DataFrame(columns=["id", "title", "overview", "genre_ids"])
    print("⚠️ movies.csv not found — empty DataFrame loaded.")

# --- Preprocess ---
movies_df = movies_df.fillna("")

# Combine features (title + overview + genres)
movies_df["combined"] = (
    movies_df["title"].astype(str)
    + " "
    + movies_df["overview"].astype(str)
    + " "
    + movies_df["genre_ids"].astype(str)
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(movies_df["combined"])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ StreamHub ML Service running (Content-Based)"})

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        title = data.get("title")

        if not title:
            return jsonify({"error": "Movie title is required"}), 400

        if title not in movies_df["title"].values:
            return jsonify({"error": f"Movie '{title}' not found in dataset."}), 404

        # Find index of the given movie
        idx = movies_df[movies_df["title"] == title].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

        # Get top 10 similar movies
        movie_indices = [i[0] for i in sim_scores]
        recommendations = movies_df.iloc[movie_indices]["title"].tolist()

        return jsonify({
            "input": title,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
