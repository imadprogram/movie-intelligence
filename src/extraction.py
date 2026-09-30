# import sys
# from pathlib import Path
# # Add project root directory to sys.path
# sys.path.append(str(Path(__file__).resolve().parent.parent))


import time
import json
import requests
from src.config import TMDB_API_KEY, TMDB_BASE_URL, DATA_RAW_DIR

def fetch_popular_movie_ids(num_pages=75):
    """
    Step 1: Fetch list of movie IDs from the /discover/movie endpoint.
    20 movies per page * 75 pages = 1,500 movies.
    """
    movie_ids = []
    endpoint = f"{TMDB_BASE_URL}/discover/movie"
    
    headers = {"accept": "application/json"}
    
    print(f"[*] Discovering movies across {num_pages} pages...")
    for page in range(1, num_pages + 1):
        params = {
            "api_key": TMDB_API_KEY,
            "sort_by": "popularity.desc",
            "vote_count.gte": 50,  # Filter out obscure movies with no votes
            "page": page
        }
        
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                for movie in results:
                    movie_ids.append(movie["id"])
            elif response.status_code == 429:
                print("[!] Rate limit reached. Sleeping 5 seconds...")
                time.sleep(5)
                continue
            else:
                print(f"[!] Warning: Page {page} returned status {response.status_code}")
        except requests.RequestException as e:
            print(f"[!] Error on page {page}: {e}")
            
        time.sleep(0.05)  # Respect API rate limits
        
    print(f"[✓] Discovered {len(movie_ids)} movie IDs.")
    return movie_ids


def fetch_movie_details(movie_id):
    """
    Step 2: Fetch detailed metadata (budget, revenue, runtime, keywords, genres).
    Uses 'append_to_response=keywords' to combine two API calls into one.
    """
    endpoint = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "keywords"
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            time.sleep(2)
            return fetch_movie_details(movie_id)  # retry once
        return None
    except requests.RequestException:
        return None


def extract_and_save(num_pages=75, output_file="movies_raw.json"):
    """
    Step 3: Main extraction pipeline. Orchestrates discovery, details, and raw JSON export.
    """
    ids = fetch_popular_movie_ids(num_pages=num_pages)
    movies_data = []
    
    print(f"[*] Fetching full details for {len(ids)} movies...")
    for index, m_id in enumerate(ids, 1):
        details = fetch_movie_details(m_id)
        if details:
            movies_data.append(details)
            
        if index % 100 == 0 or index == len(ids):
            print(f"    Progress: {index}/{len(ids)} movies fetched...")
            
        time.sleep(0.05)
        
    
    output_path = DATA_RAW_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(movies_data, f, ensure_ascii=False, indent=2)
        
    print(f"[✓] Successfully saved {len(movies_data)} raw movies to {output_path}")
    return output_path

if __name__ == "__main__":
    extract_and_save(num_pages=75)







# python -m src.extraction