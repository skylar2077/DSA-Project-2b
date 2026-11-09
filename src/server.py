import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Import your teammate's functions from cos_sim_recs.py
from cos_sim_recs import load_books, build_embeddings, recommend

print("Loading dataset and embeddings into memory...")
# Run teammate's data loader & embedding builder once on startup
BOOKS_DF = load_books()
EMBEDDINGS = build_embeddings(BOOKS_DF)
print("Data loaded! Server is ready.")

class RecommendationHandler(SimpleHTTPRequestHandler):

    def do_POST(self):
        # Handle requests sent to /api/recommend from mainpage.html
        if self.path == '/api/recommend':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode('utf-8'))

            query = payload.get('query', '')
            top_n = payload.get('top_n', 5)
            min_rating = payload.get('minimum_rating', 0.0)
            min_votes = payload.get('number_of_ratings', 0)

            try:
                # Call teammate's recommend function directly
                res_df = recommend(
                    df=BOOKS_DF,
                    emb=EMBEDDINGS,
                    query=query,
                    n=top_n,
                    min_rating=min_rating,
                    min_votes=min_votes
                )

                # Map column name to match the HTML renderer
                if "similarity" in res_df.columns:
                    res_df["similarity_score"] = res_df["similarity"]

                # Clean missing dataset values so JSON doesn't break
                res_df = res_df.fillna({
                    "img": "",
                    "isbn": "",
                    "isbn13": "",
                    "bookformat": "Paperback",
                    "rating": 0.0,
                    "pages": 0,
                    "genre": "General"
                })

                # Convert DataFrame into a list of dicts (the vector array)
                vector_results = res_df.to_dict(orient="records")

            except Exception as e:
                print(f"Search error for query '{query}': {e}")
                vector_results = []

            # Send vector back to mainpage.html
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(vector_results).encode('utf-8'))
        else:
            self.send_error(404)

    def do_GET(self):
        # Serve mainpage.html automatically at http://localhost:8000
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