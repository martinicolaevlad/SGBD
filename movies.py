import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st

def fetch_movie_data(api_key, genres=None, num_pages=5):
    base_url = "https://api.themoviedb.org/3/discover/movie"
    movies = []
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    for page in range(1, num_pages + 1):
        params = {
            "sort_by": "popularity.desc",
            "page": page,
            "with_genres": ",".join(genres) if genres else None
        }
        try:
            response = requests.get(base_url, params=params, headers=headers)
            data = response.json()
            if response.status_code == 200:
                for movie in data.get("results", []):
                    movies.append({
                        "title": movie.get("title"),
                        "popularity": movie.get("popularity"),
                        "vote_average": movie.get("vote_average"),
                        "vote_count": movie.get("vote_count"),
                        "release_date": movie.get("release_date"),
                    })
            else:
                print(f"Error: {data.get('status_message', 'Unknown error')}")
        except Exception as e:
            print(f"Error fetching data: {e}")
    return pd.DataFrame(movies)

def fetch_imdb_movie_data(num_pages=5):
    base_url = "https://www.imdb.com/search/title/?genres=action,comedy,drama&sort=year,desc"
    movies = []
    for page in range(1, num_pages + 1):
        url = f"{base_url}&start={page * 50 + 1}"
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            movie_containers = soup.find_all("div", class_="lister-item mode-advanced")
            for container in movie_containers:
                title = container.h3.a.text
                release_year = container.h3.find("span", class_="lister-item-year").text
                popularity = float(container.strong.text.split()[-1]) if container.strong else 0.0
                vote_average = float(container.find("strong").text) if container.find("strong") else 0.0
                vote_count = int(container.find("span", attrs={"name": "nv"}).text.replace(",", "")) if container.find(
                    "span", attrs={"name": "nv"}) else 0
                movies.append({
                    "title": title,
                    "popularity": popularity,
                    "vote_average": vote_average,
                    "vote_count": vote_count,
                    "release_date": release_year,
                })
        except Exception as e:
            print(f"Error fetching IMDb data: {e}")
    return pd.DataFrame(movies)

def save_to_database(df, db_name="movies.db", table_name="movies"):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

def load_from_database(db_name="movies.db", table_name="movies"):
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess_movies(df):
    df.dropna(inplace=True)
    df["popularity_scaled"] = (df["popularity"] - df["popularity"].mean()) / df["popularity"].std()
    df["vote_average_scaled"] = (df["vote_average"] - df["vote_average"].mean()) / df["vote_average"].std()
    return df

def cluster_movies(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["popularity_scaled", "vote_average_scaled"]])
    return df

def visualize_clusters(df):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df[["popularity_scaled", "vote_average_scaled"]])
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df["cluster"], cmap="viridis", s=100, alpha=0.7)
    plt.colorbar(label="Cluster")
    for i, txt in enumerate(df["title"]):
        plt.annotate(txt, (df_pca[i, 0], df_pca[i, 1]), fontsize=9, alpha=0.7)
    plt.title("Movie Clusters Based on Popularity and Ratings")
    plt.xlabel("PCA Component 1 (Popularity)")
    plt.ylabel("PCA Component 2 (Ratings)")
    plt.show()

def display_ui(df):
    st.title("Movie Recommendation and Clustering")
    st.write("Explore clusters of popular movies based on their ratings and popularity.")
    st.dataframe(df)

def main():
    try:
        print("Loading data from the local database...")
        movie_df = load_from_database()
    except Exception:
        print("No data found locally. Fetching from IMDb...")
        movie_df = fetch_imdb_movie_data()
        if movie_df.empty:
            print("No data fetched. Exiting.")
            return
        print("Saving data to the local database...")
        save_to_database(movie_df)
    print("Preprocessing movie data...")
    movie_df = preprocess_movies(movie_df)
    print("Clustering movies...")
    movie_df = cluster_movies(movie_df, n_clusters=4)
    print("Visualizing clusters...")
    visualize_clusters(movie_df)
    print("Launching Streamlit app...")
    display_ui(movie_df)

    def plot_popularity_vs_vote_average(df):
        plt.figure(figsize=(10, 6))
        plt.scatter(df["popularity"], df["vote_average"], c=df["cluster"], cmap="viridis", s=100, alpha=0.7)
        plt.title("Movie Popularity vs. Vote Average by Cluster")
        plt.xlabel("Popularity")
        plt.ylabel("Vote Average")
        plt.colorbar(label="Cluster")
        plt.show()

    plot_popularity_vs_vote_average(movie_df)

    def plot_popularity_histogram(df):
        plt.figure(figsize=(10, 6))
        plt.hist(df["popularity"], bins=30, color='skyblue', edgecolor='black')
        plt.title("Distribution of Movie Popularity")
        plt.xlabel("Popularity")
        plt.ylabel("Frequency")
        plt.show()

    plot_popularity_histogram(movie_df)

if __name__ == "__main__":
    main()
