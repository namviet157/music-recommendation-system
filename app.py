import streamlit as st
import pandas as pd

# Set page config FIRST before importing modules with cache decorators
st.set_page_config(
    page_title="Music Recommender",
    page_icon=None,
    layout="wide"
)

# Import after page config is set
from recommender import recommend_songs, compare_models, get_df, get_model

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    /* Primary button (Spotify) - emerald green */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%) !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%) !important;
        box-shadow: 0 4px 12px rgba(46, 204, 113, 0.4) !important;
    }
    /* Secondary button (YouTube) - ruby red */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #E63946 0%, #C1121F 100%) !important;
        color: white !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #EF233C 0%, #E63946 100%) !important;
        box-shadow: 0 4px 12px rgba(230, 57, 70, 0.4) !important;
    }
    .song-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1DB954;
    }
    .similarity-badge {
        background: #1DB954;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    h1 {
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.title("Music Recommendation System")
st.markdown("### Find similar songs using different AI models")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Model selection (now includes SBERT)
    model_options = {
        'tfidf': 'TF-IDF (Baseline)',
        'fasttext': 'FastText (Semantic)',
        'w2v': 'Word2Vec (Custom-trained)',
        'sbert': 'SBERT (Contextual)'
    }
    
    mode = st.radio(
        "Mode:",
        ["Single Model", "Compare All Models"],
        index=0
    )
    
    if mode == "Single Model":
        selected_model = st.selectbox(
            "Select Model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
    
    top_k = st.slider("Number of recommendations:", min_value=3, max_value=15, value=5)
    
    st.markdown("---")
    st.markdown("""
    **Model Descriptions:**
    
    **TF-IDF**: Term frequency baseline, good for exact word matching
    
    **FastText**: Custom-trained on lyrics, handles slang & misspellings
    
    **Word2Vec**: Trained directly on lyrics dataset, captures semantic relationships
    
    **SBERT**: Sentence-BERT, captures full sentence context and word order
    """)

# Main content - Cache song list to avoid recomputing
@st.cache_data
def get_song_list():
    """Get sorted list of songs (cached)."""
    df_local = get_df()
    return sorted(df_local['song'].dropna().unique())

# Load data with error handling
try:
    df = get_df()
    song_list = get_song_list()
except Exception as e:
    st.error(f"Error initializing app: {e}")
    st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    selected_song = st.selectbox(
        "Select a song:",
        song_list,
        index=0,
        help="Choose a song to find similar recommendations"
    )

# Get song info
if selected_song:
    song_info = df[df['song'] == selected_song].iloc[0]
    st.markdown(f"**Selected:** *{selected_song}* by **{song_info['artist']}**")

if st.button("Get Recommendations", type="primary"):
    
    if mode == "Compare All Models":
        # Compare all models
        st.markdown("---")
        st.subheader("Model Comparison")
        
        with st.spinner("Analyzing with all models..."):
            all_results = compare_models(selected_song, top_k=top_k)
        
        # Create columns for each model (now including SBERT)
        cols = st.columns(4)
        model_keys = ['tfidf', 'fasttext', 'w2v', 'sbert']
        
        for i, model_key in enumerate(model_keys):
            with cols[i]:
                st.markdown(f"### {model_options[model_key]}")
                
                results = all_results[model_key]
                
                if isinstance(results, str):
                    st.warning(results)
                else:
                    for _, row in results.iterrows():
                        st.markdown(f"""
                        <div class="song-card">
                            <strong>{row['song']}</strong><br>
                            <small>{row['artist']}</small><br>
                            <span class="similarity-badge">Similarity: {row['similarity']:.2%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Music links
                        link_cols = st.columns(2)
                        with link_cols[0]:
                            st.link_button("🎵 Spotify", row['spotify_url'], type="primary")
                        with link_cols[1]:
                            st.link_button("▶️ YouTube", row['youtube_url'], type="secondary")
                        st.markdown("")
    
    else:
        # Single model
        st.markdown("---")
        st.subheader(f"Recommendations using {model_options[selected_model]}")
        
        # Get model info for display (lazy loaded)
        model = get_model(selected_model)
        with st.spinner(f"Finding similar songs with {model['name']}..."):
            recommendations = recommend_songs(selected_song, selected_model, top_k=top_k)
        
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            # Display as cards
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **{idx + 1}. {row['song']}**  
                        *{row['artist']}*  
                        Similarity: `{row['similarity']:.2%}`
                        """)
                    
                    with col2:
                        st.link_button("🎵 Spotify", row['spotify_url'], type="primary")
                    
                    with col3:
                        st.link_button("▶️ YouTube", row['youtube_url'], type="secondary")
                    
                    st.divider()
            
            # Show data table
            with st.expander("View as Table"):
                display_df = recommendations[['song', 'artist', 'similarity']].copy()
                display_df['similarity'] = display_df['similarity'].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_df, hide_index=True)
