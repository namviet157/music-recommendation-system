from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import faiss
from sklearn.preprocessing import normalize
import urllib.parse

REPO_ID = "namviet157/music-recommendation"

def load_models():
    """Load all models and data from Hugging Face Hub."""
    
    # Download files from Hugging Face
    df_path = hf_hub_download(repo_id=REPO_ID, filename="df_cleaned.parquet")
    
    # TF-IDF
    tfidf_emb_path = hf_hub_download(repo_id=REPO_ID, filename="embeddings_tfidf.npz")
    tfidf_index_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_tfidf.index")
    
    # FastText
    fasttext_emb_path = hf_hub_download(repo_id=REPO_ID, filename="embeddings_fasttext.npz")
    fasttext_index_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_fasttext.index")
    
    # Word2Vec
    w2v_emb_path = hf_hub_download(repo_id=REPO_ID, filename="embeddings_w2v.npz")
    w2v_index_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_w2v.index")

    # Load DataFrame
    df = pd.read_parquet(df_path)
    
    # Load TF-IDF embeddings and normalize
    tfidf_matrix = load_npz(tfidf_emb_path)
    X_tfidf = tfidf_matrix.toarray().astype('float32')
    X_tfidf = normalize(X_tfidf, norm="l2")
    
    # Load FastText embeddings and normalize
    X_fasttext = np.load(fasttext_emb_path)["arr_0"].astype('float32')
    X_fasttext = normalize(X_fasttext, norm="l2")
    
    # Load Word2Vec embeddings and normalize
    X_w2v = np.load(w2v_emb_path)["arr_0"].astype('float32')
    X_w2v = normalize(X_w2v, norm="l2")
    
    # Load FAISS indexes
    index_tfidf = faiss.read_index(tfidf_index_path)
    index_fasttext = faiss.read_index(fasttext_index_path)
    index_w2v = faiss.read_index(w2v_index_path)

    models = {
        'tfidf': {'X': X_tfidf, 'index': index_tfidf, 'name': 'TF-IDF'},
        'fasttext': {'X': X_fasttext, 'index': index_fasttext, 'name': 'FastText'},
        'w2v': {'X': X_w2v, 'index': index_w2v, 'name': 'Word2Vec'}
    }
    
    return df, models

# Load models on import
df, models = load_models()


def get_spotify_search_url(song: str, artist: str) -> str:
    """Generate Spotify search URL for a song."""
    query = f"{song} {artist}"
    encoded_query = urllib.parse.quote(query)
    return f"https://open.spotify.com/search/{encoded_query}"


def get_youtube_search_url(song: str, artist: str) -> str:
    """Generate YouTube search URL for a song."""
    query = f"{song} {artist}"
    encoded_query = urllib.parse.quote(query)
    return f"https://www.youtube.com/results?search_query={encoded_query}"


def recommend_songs(song_name: str, model_type: str = 'tfidf', df=df, top_k: int = 5):
    """
    Recommend similar songs using specified model.
    
    Args:
        song_name: Name of the song to find similar songs for
        model_type: One of 'tfidf', 'fasttext', 'w2v'
        df: DataFrame with song data
        top_k: Number of recommendations to return
    
    Returns:
        DataFrame with recommended songs or error message
    """
    if model_type not in models:
        return f"Invalid model type. Choose from: {list(models.keys())}"
    
    # Find song index
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "Song not found in the dataset."
    idx = idx[0]
    
    # Get model components
    X = models[model_type]['X']
    index = models[model_type]['index']
    
    # Query FAISS index
    query = X[idx].reshape(1, -1).astype('float32')
    scores, indices = index.search(query, top_k + 1)
    
    # Build results DataFrame (exclude the query song itself)
    result_indices = indices[0][1:]
    results = df[['song', 'artist']].iloc[result_indices].copy()
    results['similarity'] = scores[0][1:]
    
    # Add music links
    results['spotify_url'] = results.apply(
        lambda row: get_spotify_search_url(row['song'], row['artist']), axis=1
    )
    results['youtube_url'] = results.apply(
        lambda row: get_youtube_search_url(row['song'], row['artist']), axis=1
    )
    
    return results.reset_index(drop=True)


def compare_models(song_name: str, df=df, top_k: int = 5):
    """
    Compare recommendations from all models for a song.
    
    Returns:
        Dictionary with results from each model
    """
    results = {}
    for model_type in models.keys():
        results[model_type] = recommend_songs(song_name, model_type, df, top_k)
    return results


# Backward compatibility
def recommend_songs_tfidf(song_name, df=df, top_k=5):
    return recommend_songs(song_name, 'tfidf', df, top_k)

def recommend_songs_fasttext(song_name, df=df, top_k=5):
    return recommend_songs(song_name, 'fasttext', df, top_k)

def recommend_songs_w2v(song_name, df=df, top_k=5):
    return recommend_songs(song_name, 'w2v', df, top_k)
