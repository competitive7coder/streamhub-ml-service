from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
cosine_sim = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(cosine_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

@app.route("/", methods=["GET"])
def home():
    return "StreamHub ML Service is running! Use POST /recommend."

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = data.get("user_id")
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    if user_id not in similarity_df.index:
        return jsonify({"error": f"user_id {user_id} not found"}), 404

    similar_users = similarity_df[user_id].sort_values(ascending=False).index[1:11]
    user_rated_movies = ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
    recommendations = (
        ratings_df[ratings_df["userId"].isin(similar_users) & ~ratings_df["movieId"].isin(user_rated_movies)]
        .groupby("movieId")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
