from huggingface_hub import hf_hub_download
import pandas as pd
from scipy.sparse import load_npz
import faiss

def load_models():
    df_path = hf_hub_download(
        repo_id="namviet157/music-recommendation",
        filename="df_cleaned.parquet"
    )
    X_path = hf_hub_download(
        repo_id="namviet157/music-recommendation",
        filename="tfidf_matrix.npz"
    )
    index_path = hf_hub_download(
        repo_id="namviet157/music-recommendation",
        filename="faiss.index"
    )

    df = pd.read_parquet(df_path)
    X = load_npz(X_path)
    index = faiss.read_index(index_path)

    return df, X, index

df, X, index = load_models()

def recommend_songs(song_name, df=df, top_k=5):

    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "Song not found in the dataset."
    idx = idx[0]

    query = X[idx].toarray().astype('float32')

    scores, indices = index.search(query, top_k + 1)

    return df[['song', 'artist']].iloc[indices[0][1:]]