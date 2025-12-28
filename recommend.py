import joblib

df = joblib.load('df_cleaned.pkl')
X = joblib.load('tfidf_matrix.pkl')
index = joblib.load('faiss_index.pkl')

def recommend_songs(song_name, df=df, top_k=5):

    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "Song not found in the dataset."
    idx = idx[0]

    query = X[idx].reshape(1, -1)

    scores, indices = index.search(query, top_k + 1)

    return df[['song', 'artist']].iloc[indices[0][1:]]