# Book recommender: embed each book's description, then find the nearest
# books by cosine similarity.

# Usage:
#     python cos_sim_recs.py "The Fault in Our Stars" -n 5

import argparse
import difflib
import os
import math

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Locate filepath
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CSV_PATH = os.path.join(DATA_DIR, "GoodReads_100k_books.csv")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")

MODEL_NAME = "all-MiniLM-L6-v2"

# Load books from csv into a DataFrame, removing unusable records
def load_books(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only books with a usable description and title
    df = df.dropna(subset=["title", "desc"])
    df = df[df["desc"].str.strip().str.len() > 30]
    # Drop duplicate title/author pairs (different editions)
    df = df.drop_duplicates(subset=["title", "author"])
    return df.reset_index(drop=True)

# Uses SentenceTransformer to build comparable vector embeddings for each book's description
# 30-40 minute run time on CPU to create vector embeddings of current dataset
def build_embeddings(df: pd.DataFrame, cache_path: str = EMB_PATH) -> np.ndarray:
    # Embed every description once, then cache
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if len(emb) == len(df):
            return emb
        print("Cached embeddings don't match the data; rebuilding...")
    # SentenceTransformer uses a trained language model to assign weights to each token
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        df["desc"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, emb)
    return emb

# searches for a matching title within the DataFrame
def find_book_index(df: pd.DataFrame, query: str) -> int:
    #Exact (case-insensitive) match
    titles = df["title"].str.lower()
    q = query.lower().strip()

    exact = np.where(titles == q)[0]
    if len(exact):
        return int(exact[0])

    raise ValueError(f"No book found matching '{query}'")

# Calculates the dot product of two 384 size vectors
def dot(a, b) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total

# Calculates the normal value of a vector
def norm(a) -> float:
    return math.sqrt(dot(a, a))

# Calculates cosine similarity by dividing the dot product of vectors A and B by |A|*|B|
def cosine_similarity(a, b) -> float:
    denom = norm(a) * norm(b)
    if denom == 0: # avoid dividing by zero for an all-zero vector
        return 0.0
    return dot(a, b) / denom

# Calculates cosine similarity comparing the target book with the dataset, returning a list of scores
def similarities(emb, idx) -> list[float]:
    query = emb[idx]
    return [cosine_similarity(query, row) for row in emb]

# Finds similar books to target book using cosine similarity, with filtering functionality
def recommend(
    df: pd.DataFrame,
    emb: np.ndarray,
    query: str,
    n: int = 5,
    min_rating: float = 0.0,
    min_votes: int = 0,
) -> pd.DataFrame:
    idx = find_book_index(df, query)
    print(f"Matched: {df.loc[idx, 'title']} by {df.loc[idx, 'author']}\n")
 
    sims = similarities(emb, idx)
 
    # Rating filter
    passes = (df["rating"] >= min_rating) & (df["totalratings"] >= min_votes)
    # books that don't pass filter set similiarity score to -1
    sims = np.where(passes.to_numpy(), sims, -1.0)
    sims[idx] = -1.0  # never recommend the given query book

    # Table is sorted descending from similiarity score
    top = np.argsort(sims)[::-1][:n]
    top = [i for i in top if sims[i] > -1.0]  # drop slots left empty by filters
    # relevant data is sliced from top book entries
    out = df.loc[top, ["title", "desc", "genre", "img", "isbn", "link", "pages", "rating", "reviews", "title", "totalratings"]].copy()
    out["similarity"] = sims[top].round(3) # similarity score column is added to final output
    return out.reset_index(drop=True)

def convert_matrix(df: pd.DataFrame) -> list[list[float]]:
    return df.to_numpy()

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend similar books.")
    parser.add_argument("title", help="Title of a book you like")
    parser.add_argument("-n", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--min-rating", type=float, default=0.0,
                        help="Minimum average rating (e.g. 3.8)")
    parser.add_argument("--min-votes", type=int, default=0,
                        help="Minimum number of ratings (e.g. 1000)")
    args = parser.parse_args()
 
    books = load_books()
    embeddings = build_embeddings(books)
    result = recommend(books, embeddings, args.title, args.n,
                       args.min_rating, args.min_votes)
    result_matrix = result.to_numpy()
    print(result.to_string())