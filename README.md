# Lyrics-Based Music Recommender

A content-based music recommendation system that suggests similar songs by analyzing lyrics similarity. Built with TF-IDF vectorization and Facebook's FAISS library for efficient nearest-neighbor search.

## Live Demo

**Try it now:** [https://huggingface.co/spaces/namviet157/music-recommendation-system](https://huggingface.co/spaces/namviet157/music-recommendation-system)

## Features

- **Lyrics-based similarity matching** - Find songs with similar lyrical content
- **Fast search with FAISS indexing** - Instant recommendations from 57,000+ songs
- **Interactive web interface** - User-friendly Streamlit application
- **Cloud-hosted models** - Pre-trained models stored on Hugging Face Hub

## Demo

The web application allows you to:
1. Select a song from the dropdown menu
2. Click "Recommend Similar Songs"
3. Get instant recommendations based on lyrics similarity

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Text Processing** | NLTK, scikit-learn |
| **Feature Extraction** | TF-IDF Vectorization |
| **Similarity Search** | FAISS (Facebook AI Similarity Search) |
| **Model Storage** | Hugging Face Hub |
| **Deployment** | Streamlit Cloud |
| **Data** | Spotify Million Song Dataset |

## Project Structure

```
music-recommendation-system/
├── app.py                             # Streamlit web application
├── recommend.py                       # Recommendation logic module
├── Music_Recommendation_System.ipynb  # Training notebook
├── requirements.txt                   # Python dependencies
└── data/
    └── spotify_millsongdata.csv       # Dataset (download required)
```

**Model files (hosted on Hugging Face Hub):**
- `df_cleaned.parquet` - Preprocessed DataFrame
- `faiss.index` - FAISS index for similarity search  
- `tfidf_matrix.npz` - TF-IDF sparse matrix

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/namviet157/music-recommendation-system.git
   cd music-recommendation-system
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Run Locally (Models auto-download from Hugging Face)

```bash
streamlit run app.py
```

The app will automatically download pre-trained models from [Hugging Face Hub](https://huggingface.co/namviet157/music-recommendation) and open in your browser at `http://localhost:8501`

#### Option 2: Train from Scratch

1. **Set up Kaggle API**
   - Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
   - Go to Settings → API → Create New Token
   - Save `kaggle.json` in the project root

2. **Run the training notebook**
   - Open `Music_Recommendation_System.ipynb` in Jupyter
   - Run all cells to download data and train models

3. **Launch the app**
   ```bash
   streamlit run app.py
   ```

## How It Works

### 1. Text Preprocessing
- Remove special characters and convert to lowercase
- Tokenize lyrics into individual words
- Remove common stopwords (the, is, at, etc.)

### 2. TF-IDF Vectorization
- Convert cleaned lyrics to numerical vectors
- TF-IDF weights words by importance (frequent in song, rare across dataset)
- Limited to 5,000 features for efficiency

### 3. FAISS Indexing
- L2 normalize vectors for cosine similarity
- Build FAISS IndexFlatIP for exact inner product search
- Enables millisecond-level similarity queries

### 4. Recommendation
- Look up the selected song's vector
- Find k-nearest neighbors using FAISS
- Return most similar songs based on lyrics

## Dataset

- **Source**: [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
- **Size**: 57,650 songs
- **Features**: Artist, Song Title, Lyrics, Link

## Configuration

You can modify these parameters in the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_features` | 5000 | Number of TF-IDF features |
| `top_k` | 5 | Number of recommendations |

## API Reference

### `recommend_songs(song_name, df, top_k=5)`

Returns similar songs based on lyrics.

**Parameters:**
- `song_name` (str): Name of the song to find similar songs for
- `df` (DataFrame): Dataset with song information
- `top_k` (int): Number of recommendations to return

**Returns:**
- DataFrame with columns: `song`, `artist`

## Resources

- **Hugging Face Models**: [namviet157/music-recommendation](https://huggingface.co/namviet157/music-recommendation)
- **Live App**: [Streamlit Cloud](https://music-recommendation-system-mk7huqfthpgkbkpcdtafgj.streamlit.app/)
- **Dataset**: [Kaggle - Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
