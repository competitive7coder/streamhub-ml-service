from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)

# --- Load movies data safely ---
try:
    movies_df = pd.read_csv("movies.csv")
    print("✅ movies.csv loaded successfully.")
except Exception as e:
    print("❌ Error loading movies.csv:", e)
    movies_df = pd.DataFrame(columns=["id", "title"])

# --- Ensure essential columns exist ---
for col in ["id", "title", "overview", "genre_ids"]:
    if col not in movies_df.columns:
        movies_df[col] = ""

movies_df = movies_df.fillna("")

# --- Combine available text features ---
movies_df["combined"] = (
    movies_df["title"].astype(str)
    + " "
    + movies_df["overview"].astype(str)
    + " "
    + movies_df["genre_ids"].astype(str)
)

# --- TF-IDF Vectorization ---
if not movies_df.empty:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(movies_df["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
else:
    cosine_sim = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ StreamHub ML Service running (Content-Based, no ratings)"})

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        title = data.get("title")

        if not title:
            return jsonify({"error": "Movie title is required"}), 400

        if title not in movies_df["title"].values:
            return jsonify({"error": f"Movie '{title}' not found in dataset."}), 404

        if cosine_sim is None:
            return jsonify({"error": "Similarity matrix not built — empty dataset."}), 500

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
