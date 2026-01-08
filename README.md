# Lyrics-Based Music Recommender

A content-based music recommendation system that suggests similar songs by analyzing lyrics similarity. Features multiple embedding models (TF-IDF, FastText, Word2Vec, SBERT) with Facebook's FAISS library for efficient nearest-neighbor search.

## Live Demo

**Try it now:** [https://huggingface.co/spaces/namviet157/music-recommendation-system](https://huggingface.co/spaces/namviet157/music-recommendation-system)

## Features

- **Multiple embedding models** - Compare TF-IDF, FastText, Word2Vec, and SBERT models
- **Lyrics-based similarity matching** - Find songs with similar lyrical content
- **Fast search with FAISS indexing** - Instant recommendations from 57,000+ songs
- **Model comparison mode** - View recommendations from all models side-by-side
- **Interactive web interface** - User-friendly Streamlit application with modern UI
- **Cloud-hosted models** - Pre-trained models stored on Hugging Face Hub
- **Memory optimized** - Lazy loading and caching for efficient resource usage

## Demo

The web application allows you to:
1. Select a song from the dropdown menu
2. Choose a single model or compare all models
3. Adjust the number of recommendations (3-15)
4. Click "Get Recommendations" to see similar songs
5. View results with similarity scores and direct links to Spotify and YouTube


## Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Text Processing** | NLTK, scikit-learn |
| **Feature Extraction** | TF-IDF, FastText, Word2Vec, Sentence-BERT |
| **Embedding Models** | scikit-learn, Gensim, sentence-transformers |
| **Similarity Search** | FAISS (Facebook AI Similarity Search) |
| **Model Storage** | Hugging Face Hub |
| **Deployment** | Hugging Face Spaces |
| **Data** | Spotify Million Song Dataset |

## Project Structure

```
music-recommendation-system/
├── app.py                             # Streamlit web application
├── recommender.py                     # Recommendation logic module
├── requirements.txt                   # Python dependencies
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_pipeline.ipynb        # Data preprocessing
│   ├── 02_modeling_and_recommendation.ipynb  # Model training
│   └── 03_evaluation_and_deployment.ipynb    # Evaluation & deployment
├── data/
│   ├── raw/                          # Raw dataset
│   ├── processed/                    # Processed data
│   └── embeddings/                   # Saved embeddings (local)
└── models/
    └── faiss_indexes/                # FAISS indexes (local)
```

**Model files (hosted on Hugging Face Hub):**
- `df_cleaned.parquet` - Preprocessed DataFrame
- `embeddings_tfidf.npz` - TF-IDF embeddings
- `embeddings_fasttext.npz` - FastText embeddings
- `embeddings_w2v.npz` - Word2Vec embeddings
- `embeddings_sbert.npz` - SBERT embeddings
- `faiss_tfidf.index` - FAISS index for TF-IDF
- `faiss_fasttext.index` - FAISS index for FastText
- `faiss_w2v.index` - FAISS index for Word2Vec
- `faiss_sbert.index` - FAISS index for SBERT

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

2. **Run the training notebooks**
   - Open `notebooks/01_data_pipeline.ipynb` - Preprocess the dataset
   - Open `notebooks/02_modeling_and_recommendation.ipynb` - Train all models
   - Open `notebooks/03_evaluation_and_deployment.ipynb` - Evaluate and deploy

3. **Launch the app**
   ```bash
   streamlit run app.py
   ```

## How It Works

### 1. Text Preprocessing
- Remove special characters and convert to lowercase
- Tokenize lyrics into individual words
- Remove common stopwords (the, is, at, etc.)
- Handle contractions and normalize text

### 2. Embedding Generation
The system supports four different embedding approaches:

- **TF-IDF**: Term frequency-inverse document frequency baseline, good for exact word matching
- **FastText**: Custom-trained on lyrics dataset, handles slang and misspellings with subword information
- **Word2Vec**: Trained directly on lyrics dataset, captures semantic relationships between words
- **SBERT**: Sentence-BERT model, captures full sentence context and word order

### 3. FAISS Indexing
- L2 normalize all embedding vectors for cosine similarity
- Build FAISS IndexFlatIP for exact inner product search
- Separate indexes for each model type
- Enables millisecond-level similarity queries

### 4. Recommendation
- Look up the selected song's embedding vector
- Find k-nearest neighbors using FAISS
- Return most similar songs with similarity scores
- Support for single model or multi-model comparison

## Dataset

- **Source**: [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
- **Size**: 57,650 songs
- **Features**: Artist, Song Title, Lyrics, Link

## Configuration

You can modify these parameters in the app interface:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 5 | Number of recommendations (3-15) |
| `model` | TF-IDF | Choose from: TF-IDF, FastText, Word2Vec, SBERT |

Model-specific parameters can be adjusted in `notebooks/02_modeling_and_recommendation.ipynb`:
- TF-IDF: `max_features` (default: 5000)
- FastText: `vector_size`, `window`, `min_count`
- Word2Vec: `vector_size`, `window`, `min_count`
- SBERT: Model name (default: `all-MiniLM-L6-v2`)

## API Reference

### `recommend_songs(song_name, model_type='tfidf', top_k=5)`

Returns similar songs based on lyrics using the specified model.

**Parameters:**
- `song_name` (str): Name of the song to find similar songs for
- `model_type` (str): Model to use - 'tfidf', 'fasttext', 'w2v', or 'sbert'
- `top_k` (int): Number of recommendations to return

**Returns:**
- DataFrame with columns: `song`, `artist`, `similarity`, `spotify_url`, `youtube_url`

### `compare_models(song_name, top_k=5)`

Returns recommendations from all models for comparison.

**Parameters:**
- `song_name` (str): Name of the song to find similar songs for
- `top_k` (int): Number of recommendations per model

**Returns:**
- Dictionary with keys: 'tfidf', 'fasttext', 'w2v', 'sbert'
- Each value is a DataFrame with recommendations from that model

## Resources

- **Live App**: [Hugging Face Spaces](https://huggingface.co/spaces/namviet157/music-recommendation-system)
- **Hugging Face Models**: [namviet157/music-recommendation](https://huggingface.co/namviet157/music-recommendation)
- **Dataset**: [Kaggle - Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)

## Model Details

### TF-IDF (Baseline)
- **Type**: Term frequency-inverse document frequency
- **Best for**: Exact word matching, baseline comparison
- **Training**: scikit-learn TfidfVectorizer

### FastText (Semantic)
- **Type**: Custom-trained FastText embeddings
- **Best for**: Handling slang, misspellings, and subword information
- **Training**: Gensim FastText on lyrics dataset

### Word2Vec (Semantic)
- **Type**: Custom-trained Word2Vec embeddings
- **Best for**: Capturing semantic relationships between words
- **Training**: Gensim Word2Vec on lyrics dataset (not pre-trained Google News)

### SBERT (Contextual)
- **Type**: Sentence-BERT embeddings
- **Best for**: Full sentence context and word order understanding
- **Model**: `all-MiniLM-L6-v2` from sentence-transformers
