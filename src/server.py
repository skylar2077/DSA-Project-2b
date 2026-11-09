import json
import os
import pickle
import string
import numpy as np
import pandas as pd
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Import Cosine engine from cos_sim_recs.py
from cos_sim_recs import load_books, build_embeddings, recommend as recommend_cosine

# Try loading BM25 engine & pickle data
HAS_BM25 = False
BM25_PICKLE = None

try:
    from bm25section import bm25O
    
    # Path to data.pkl (located in ../data/data.pkl relative to src/)
    pickle_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            BM25_PICKLE = pickle.load(f)
        HAS_BM25 = True
        print("BM25 pickle loaded successfully!")
    else:
        print(f"Warning: BM25 pickle not found at {pickle_path}. BM25 requests will fall back to Cosine search.")
except Exception as e:
    print(f"Warning: BM25 initialization issue ({e}). Defaulting to Cosine search.")

print("Loading dataset into memory...")
BOOKS_DF = load_books()
EMBEDDINGS = build_embeddings(BOOKS_DF)
print("Data loaded! Server is ready.")


def recommend_bm25(df, query_str, top_n, min_rating, min_votes, weights):
    # Pass the raw query string directly to getScores()
    raw_query = str(query_str)

    bm25_weights = {
        "title": weights.get("title", 0.25),
        "authors": weights.get("author", 0.25),
        "descs": weights.get("desc", 0.25),
        "genres": weights.get("genre", 0.25)
    }

    titleVec = BM25_PICKLE[0].getScores(raw_query)
    authorsVec = BM25_PICKLE[1].getScores(raw_query)
    descsVec = BM25_PICKLE[2].getScores(raw_query)
    genresVec = BM25_PICKLE[3].getScores(raw_query)

    num_items = min(len(df), len(titleVec))
    combined_scores = np.zeros(num_items)

    for i in range(num_items):
        combined_scores[i] = (
            titleVec[i] * bm25_weights["title"] +
            authorsVec[i] * bm25_weights["authors"] +
            descsVec[i] * bm25_weights["descs"] +
            genresVec[i] * bm25_weights["genres"]
        )

    df_copy = df.iloc[:num_items].copy()
    df_copy["raw_bm25"] = combined_scores

    # Filter dataset based on minimum ratings & count
    if "rating" in df_copy.columns:
        df_copy = df_copy[df_copy["rating"] >= min_rating]
    if "totalratings" in df_copy.columns:
        df_copy = df_copy[df_copy["totalratings"] >= min_votes]
    elif "rating_count" in df_copy.columns:
        df_copy = df_copy[df_copy["rating_count"] >= min_votes]

    # Filter out items with no matching terms
    df_copy = df_copy[df_copy["raw_bm25"] > 0]

    # Sort descending and return top_n items
    top_df = df_copy.sort_values(by="raw_bm25", ascending=False).head(top_n).copy()

    # Normalize column names for frontend rendering
    column_mapping = {
        "authors": "author",
        "descs": "desc",
        "genres": "genre",
        "reviews": "rating_count",
        "totalratings": "rating_count"
    }
    
    # BM25 does not use normalized similarity scores
    top_df["similarity_score"] = None
    
    return top_df.rename(columns=column_mapping)


class RecommendationHandler(SimpleHTTPRequestHandler):

    def do_POST(self):
        if self.path == '/api/recommend':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode('utf-8'))

            query = payload.get('query', '')
            algo = payload.get('algorithm', 'bm25')  # 'bm25' or 'cosine'
            top_n = payload.get('top_n', 5)
            min_rating = payload.get('minimum_rating', 0.0)
            min_votes = payload.get('number_of_ratings', 0)
            weights = payload.get('weights', {})

            try:
                if algo == 'bm25' and HAS_BM25:
                    res_df = recommend_bm25(
                        df=BOOKS_DF,
                        query_str=query,
                        top_n=top_n,
                        min_rating=min_rating,
                        min_votes=min_votes,
                        weights=weights
                    )
                else:
                    res_df = recommend_cosine(
                        df=BOOKS_DF,
                        emb=EMBEDDINGS,
                        query=query,
                        n=top_n,
                        min_rating=min_rating,
                        min_votes=min_votes
                    )
                    if "similarity" in res_df.columns:
                        res_df["similarity_score"] = res_df["similarity"]

                # Clean NaN values for valid JSON response
                res_df = res_df.fillna({
                    "img": "",
                    "isbn": "",
                    "isbn13": "",
                    "bookformat": "Paperback",
                    "rating": 0.0,
                    "pages": 0,
                    "genre": "General",
                    "desc": ""
                })

                # Ensure None / null values stay null for missing scores
                res_df = res_df.where(pd.notnull(res_df), None)

                vector_results = res_df.to_dict(orient="records")

            except Exception as e:
                print(f"Error processing {algo} search: {e}")
                vector_results = []

            # Send response back to mainpage.html
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            try:
                self.wfile.write(json.dumps(vector_results).encode('utf-8'))
            except (ConnectionAbortedError, BrokenPipeError):
                print("Client disconnected before write completed.")
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == '/' or self.path == '/mainpage.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            html_path = os.path.join(current_dir, 'mainpage.html')
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            super().do_GET()


if __name__ == '__main__':
    server_address = ('127.0.0.1', 8000)
    httpd = HTTPServer(server_address, RecommendationHandler)
    print("Serving DSA Book Recommender at http://127.0.0.1:8000")
    httpd.serve_forever()