import requests
import csv
import time

#  TMDB API KEY HERE
API_KEY = "9327062a11f187243d85ffdc900311d5"# ---------------

BASE_URL = "https://api.themoviedb.org/3"
CSV_FILE = "movies.csv"

# How many pages of "popular" movies to fetch (20 movies per page)
TOTAL_PAGES_TO_FETCH = 50 # This will get 1000 movies

# 1. Get the master list of all genre IDs and their names
def get_genre_map():
    print("Fetching genre map...")
    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        genres = response.json().get('genres', [])
        # Create a dictionary for easy lookup: {12: "Adventure", 14: "Fantasy", ...}
        genre_map = {genre['id']: genre['name'] for genre in genres}
        print("Genre map fetched successfully.")
        return genre_map
    except requests.RequestException as e:
        print(f"Error fetching genres: {e}")
        return None

# 2. Fetch popular movies page by page
def fetch_popular_movies(genre_map):
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the CSV header
        writer.writerow(["id", "title", "genres"])
        
        print(f"Starting to fetch {TOTAL_PAGES_TO_FETCH} pages of popular movies...")
        
        movie_count = 0
        # Loop from page 1 up to (and including) TOTAL_PAGES_TO_FETCH
        for page in range(1, TOTAL_PAGES_TO_FETCH + 1):
            url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&page={page}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                movies = response.json().get('results', [])
                
                if not movies:
                    print("No more movies found.")
                    break # Stop if we run out of pages
                
                for movie in movies:
                    movie_id = movie.get('id')
                    title = movie.get('title', '')
                    genre_ids = movie.get('genre_ids', [])
                    
                    # Convert the list of IDs [28, 12] into a string "Action;Adventure"
                    # Use the map we fetched earlier
                    genre_names = [genre_map.get(gid) for gid in genre_ids if genre_map.get(gid)]
                    genres_str = ";".join(genre_names)
                    
                    # Write the row to the CSV file
                    writer.writerow([movie_id, title, genres_str])
                    movie_count += 1
                    
                print(f"Page {page}/{TOTAL_PAGES_TO_FETCH} processed. Total movies: {movie_count}")
                
                # Be nice to the API - wait a little
                time.sleep(0.1) 
            
            except requests.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                time.sleep(1) # Wait longer on error
                
    print(f"\nDone! Successfully saved {movie_count} movies to {CSV_FILE}.")

# 3. Main execution
if __name__ == "__main__":
    if API_KEY == "YOUR_TMDB_API_KEY_HERE":
        print("="*50)
        print("ERROR: Please open fetch_data.py and")
        print("       replace 'YOUR_TMDB_API_KEY_HERE' with your")
        print("       actual TMDB API key.")
        print("="*50)
    else:
        genres = get_genre_map()
        if genres:
            fetch_popular_movies(genres)