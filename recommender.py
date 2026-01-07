from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import faiss
from sklearn.preprocessing import normalize
import urllib.parse
import streamlit as st

REPO_ID = "namviet157/music-recommendation"

@st.cache_resource
def load_dataframe():
    """Load DataFrame with caching."""
    df_path = hf_hub_download(repo_id=REPO_ID, filename="df_cleaned.parquet")
    return pd.read_parquet(df_path)

@st.cache_resource
def load_model(model_type: str):
    """Load a single model lazily with caching."""
    if model_type == 'tfidf':
        tfidf_emb_path = hf_hub_download(repo_id=REPO_ID, filename="embeddings_tfidf.npz")
        tfidf_index_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_tfidf.index")
        
        # Keep TF-IDF as sparse matrix for memory efficiency
        tfidf_matrix = load_npz(tfidf_emb_path)
        # Only convert to dense when needed for FAISS query
        X_tfidf = tfidf_matrix.astype('float32')
        X_tfidf_dense = normalize(X_tfidf.toarray(), norm="l2")
        index_tfidf = faiss.read_index(tfidf_index_path)
        
        return {'X': X_tfidf_dense, 'index': index_tfidf, 'name': 'TF-IDF', 'sparse': X_tfidf}
    
    elif model_type == 'fasttext':
        fasttext_emb_path = hf_hub_download(repo_id=REPO_ID, filename="embeddings_fasttext.npz")
        fasttext_index_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_fasttext.index")
        
        X_fasttext = np.load(fasttext_emb_path)["arr_0"].astype('float32')
        X_fasttext = normalize(X_fasttext, norm="l2")
        index_fasttext = faiss.read_index(fasttext_index_path)
        
        return {'X': X_fasttext, 'index': index_fasttext, 'name': 'FastText'}
    
    elif model_type == 'w2v':
        w2v_emb_path = hf_hub_download(repo_id=REPO_ID, filename="embeddings_w2v.npz")
        w2v_index_path = hf_hub_download(repo_id=REPO_ID, filename="faiss_w2v.index")
        
        X_w2v = np.load(w2v_emb_path)["arr_0"].astype('float32')
        X_w2v = normalize(X_w2v, norm="l2")
        index_w2v = faiss.read_index(w2v_index_path)
        
        return {'X': X_w2v, 'index': index_w2v, 'name': 'Word2Vec'}
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

@st.cache_resource
def get_all_models():
    """Get all models dictionary (cached)."""
    return {
        'tfidf': load_model('tfidf'),
        'fasttext': load_model('fasttext'),
        'w2v': load_model('w2v')
    }

# Public API functions
def get_df():
    """Get DataFrame (cached)."""
    return load_dataframe()

def get_models():
    """Get models dictionary (cached)."""
    return get_all_models()


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


def recommend_songs(song_name: str, model_type: str = 'tfidf', df=None, top_k: int = 5):
    """
    Recommend similar songs using specified model.
    
    Args:
        song_name: Name of the song to find similar songs for
        model_type: One of 'tfidf', 'fasttext', 'w2v'
        df: DataFrame with song data (if None, will load cached version)
        top_k: Number of recommendations to return
    
    Returns:
        DataFrame with recommended songs or error message
    """
    if df is None:
        df = get_df()
    
    models = get_models()
    
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


def compare_models(song_name: str, df=None, top_k: int = 5):
    """
    Compare recommendations from all models for a song.
    
    Returns:
        Dictionary with results from each model
    """
    if df is None:
        df = get_df()
    models = get_models()
    
    results = {}
    for model_type in models.keys():
        results[model_type] = recommend_songs(song_name, model_type, df, top_k)
    return results


# Backward compatibility
def recommend_songs_tfidf(song_name, df=None, top_k=5):
    return recommend_songs(song_name, 'tfidf', df, top_k)

def recommend_songs_fasttext(song_name, df=None, top_k=5):
    return recommend_songs(song_name, 'fasttext', df, top_k)

def recommend_songs_w2v(song_name, df=None, top_k=5):
    return recommend_songs(song_name, 'w2v', df, top_k)
