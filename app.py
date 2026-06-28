import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configurations
st.set_page_config(
    page_title="AccentMark — Accent Fingerprint Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Dark Mode Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    /* Background and typography styling */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #151821 100%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3, .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Glassmorphism card structures */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.95rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .metric-desc {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }
    
    /* Speaker info box */
    .speaker-box {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Define speakers dict consistent with research metadata
SPEAKERS = {
    'ABA': 'Arabic', 'YBAA': 'Arabic',
    'SVBI': 'Hindi', 'TNI': 'Hindi',
    'HJK': 'Korean', 'YDCK': 'Korean',
    'BWC': 'Mandarin', 'LXC': 'Mandarin',
    'EBVS': 'Spanish', 'ERMS': 'Spanish',
    'HQTV': 'Vietnamese', 'PNV': 'Vietnamese',
    'NJS': 'Native', 'TXHC': 'Native'
}

RESULTS_DIR = 'results'
metrics_json_path = os.path.join(RESULTS_DIR, 'metrics.json')
means_pkl_path = os.path.join(RESULTS_DIR, 'all_means.pkl')

# Gracefully check for file existence
try:
    with open(metrics_json_path, 'r') as f:
        metrics = json.load(f)
    with open(means_pkl_path, 'rb') as f:
        all_means = pickle.load(f)
    
    # Assert generated image presence
    assert os.path.exists(os.path.join(RESULTS_DIR, 'umap_clusters.png'))
    assert os.path.exists(os.path.join(RESULTS_DIR, 'phoneme_heatmap.png'))
    assert os.path.exists(os.path.join(RESULTS_DIR, 'fingerprint_comparison.png'))
except (FileNotFoundError, AssertionError):
    st.error("### ⚠️ AccentMark Artifacts Not Found")
    st.write(
        "The metrics or visualization results could not be found. "
        "Please run the modeling and extraction notebooks first to process the audio data:"
    )
    st.info(
        "1. Open and run all cells in `notebooks/01_feature_extraction.ipynb`\n"
        "2. Open and run all cells in `notebooks/02_modeling_and_viz.ipynb`"
    )
    st.stop()

# Reconstruct Native Reference from variables
njs_means = all_means.get('NJS', {})
txhc_means = all_means.get('TXHC', {})
ref_phonemes = set(njs_means.keys()).union(txhc_means.keys())
reference_means = {}
for ph in ref_phonemes:
    if ph in njs_means and ph in txhc_means:
        reference_means[ph] = (njs_means[ph] + txhc_means[ph]) / 2.0
    elif ph in njs_means:
        reference_means[ph] = njs_means[ph]
    else:
        reference_means[ph] = txhc_means[ph]

# Sidebar Setup
st.sidebar.markdown("<h2 style='text-align: center;'>AccentMark 🎙️</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    "Most models tell you *what* accent someone has. "
    "**AccentMark** constructs a continuous acoustic fingerprint showing *exactly how* "
    "their pronunciation deviates from a native English baseline phoneme-by-phoneme."
)
st.sidebar.markdown("---")

# Speaker Selector dropdown
non_native_spk = [s for s, l1 in SPEAKERS.items() if l1 != 'Native']
spk_options = [f"{spk} ({SPEAKERS[spk]} L1)" for spk in non_native_spk]
selected_option = st.sidebar.selectbox("Select Target Speaker to Analyze", spk_options)
selected_speaker = selected_option.split(" ")[0]

st.sidebar.markdown("---")
st.sidebar.markdown("### L2-ARCTIC Details")
st.sidebar.markdown(
    "- **Target Languages**: Arabic, Hindi, Korean, Mandarin, Spanish, Vietnamese\n"
    "- **Reference Speakers**: CMU ARCTIC Native Speakers (NJS, TXHC)\n"
    "- **Features**: Durations, RMS Energy, Spectral Centroids, ZCR, 13 MFCCs per phoneme."
)

# Header Section
st.markdown("<h1 style='font-size: 2.8rem; margin-bottom: 5px;'>AccentMark — Accent Fingerprint Explorer</h1>", unsafe_allow_html=True)
st.markdown("### Continuous acoustic deviation profiles at the individual phoneme level.")
st.markdown("---")

# Create Tabs for layout separation
tab1, tab2, tab3 = st.tabs(["🎙️ Speaker Analysis", "📊 Space & Clusters", "🏆 System Performance"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### Speaker Profile: {selected_speaker}")
        st.markdown(
            f"<div class='speaker-box'>"
            f"**Speaker ID**: `{selected_speaker}`<br>"
            f"**L1 Background**: `{SPEAKERS[selected_speaker]}`<br>"
            f"**Aligned Phonemes**: `{metrics['n_common_phonemes']}` phonemes common across all speakers."
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Display selected speaker's phoneme deviations in a table
        st.markdown("#### Phoneme Deviation Magnitudes")
        
        common_phonemes = sorted(list(reference_means.keys()))
        # Ensure we intersect common_phonemes with selected speaker's phonemes
        common_phonemes = [p for p in common_phonemes if p in all_means[selected_speaker]]
        
        dev_list = []
        for ph in common_phonemes:
            dev_vec = all_means[selected_speaker][ph] - reference_means[ph]
            dev_norm = np.linalg.norm(dev_vec)
            dev_list.append({"Phoneme": ph, "Deviation Magnitude": float(dev_norm)})
            
        df_dev = pd.DataFrame(dev_list).sort_values(by="Deviation Magnitude", ascending=False)
        st.dataframe(
            df_dev, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Deviation Magnitude": st.column_config.NumberColumn(format="%.3f")
            }
        )

    with col2:
        st.markdown("#### Interactive Deviation Heatmap Row")
        st.write("Below is the phonetic deviation heatmap matrix with the selected speaker highlighted.")
        
        # Reconstruct full deviation matrix for display
        heatmap_rows = []
        for spk in non_native_spk:
            row_vals = []
            for ph in common_phonemes:
                dev = all_means[spk][ph] - reference_means[ph]
                row_vals.append(np.linalg.norm(dev))
            heatmap_rows.append(row_vals)
            
        df_heatmap = pd.DataFrame(
            heatmap_rows,
            index=non_native_spk,
            columns=common_phonemes
        )
        
        # Stylize dataframe to highlight the active row
        def highlight_active(row):
            if row.name == selected_speaker:
                return ['background-color: rgba(56, 189, 248, 0.25); border: 2px solid #38bdf8; font-weight: bold'] * len(row)
            return [''] * len(row)
            
        styled_df = df_heatmap.style.apply(highlight_active, axis=1).format(precision=3)
        st.dataframe(styled_df, use_container_width=True, height=450)
        
        # Individual speaker plot
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#0f172a')
        
        # Sort values to align with dataframe display
        df_sorted_plot = df_dev.sort_values(by="Phoneme")
        
        bars = ax.bar(
            df_sorted_plot["Phoneme"], 
            df_sorted_plot["Deviation Magnitude"], 
            color="#38bdf8", 
            edgecolor="#e2e8f0", 
            alpha=0.8
        )
        
        ax.set_title(f"Pronunciation Deviation Vector Magnitude per Phoneme for {selected_speaker}", color="#e2e8f0", fontsize=12, pad=10)
        ax.set_xlabel("Phoneme", color="#94a3b8")
        ax.set_ylabel("Deviation (L2 Norm)", color="#94a3b8")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
        
        # Add values on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.0, 
                yval + 0.02, 
                f"{yval:.2f}", 
                ha='center', 
                va='bottom', 
                color='#e2e8f0', 
                fontsize=8
            )
            
        st.pyplot(fig)

with tab2:
    st.markdown("### UMAP Dimensionality Space & Dialect Projections")
    st.write(
        "By projecting the 170+ dimensional phonetic deviation vectors into low-dimensional space, "
        "we can visualize how accent characteristics emerge as continuous manifolds clustering naturally by L1 language family."
    )
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.markdown("#### UMAP Dialect Clustering")
        st.image(os.path.join(RESULTS_DIR, 'umap_clusters.png'), use_container_width=True)
        
    with col_vis2:
        st.markdown("#### Phoneme Deviation Heatmap (Full)")
        st.image(os.path.join(RESULTS_DIR, 'phoneme_heatmap.png'), use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### Multi-Speaker Fingerprint Comparison")
    st.image(os.path.join(RESULTS_DIR, 'fingerprint_comparison.png'), use_container_width=True)

with tab3:
    st.markdown("### System Diagnostics & Evaluation Metrics")
    st.write(
        "AccentMark evaluates the accent fingerprint based on clustering purity, speaker verification stability, "
        "and similarity preservation."
    )
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val'>{metrics['silhouette_score']:.3f}</div>"
            f"<div class='metric-lbl'>Silhouette Score</div>"
            f"<div class='metric-desc'>Measures the clustering quality of speakers by L1 background in the UMAP space. Scores above 0.3 demonstrate strong L1 structure.</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val'>{metrics['distance_ratio']:.2f}x</div>"
            f"<div class='metric-lbl'>L1 Cosine Distance Ratio</div>"
            f"<div class='metric-desc'>Ratio of average distance between different L1 speakers to average distance between same L1 speakers. Values > 1.0 indicate similarity alignment.</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    with col_m2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val'>{metrics['speaker_reid_accuracy'] * 100:.1f}%</div>"
            f"<div class='metric-lbl'>Speaker Re-Identification</div>"
            f"<div class='metric-desc'>Accuracy of leave-one-out 1-NN speaker matching on split session fingerprints. Evaluates if the fingerprint is stable across sessions.</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val'>{metrics['pca_variance_explained'] * 100:.1f}%</div>"
            f"<div class='metric-lbl'>PCA Variance Explained</div>"
            f"<div class='metric-desc'>Total explained variance of the top PCA components. Shows the compactness of our linear subspace projections.</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    with col_m3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val'>{metrics['n_common_phonemes']}</div>"
            f"<div class='metric-lbl'>Common Phonemes</div>"
            f"<div class='metric-desc'>The count of distinct phonetic units aligned across all speakers and references.</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val'>{metrics['fingerprint_dim']}</div>"
            f"<div class='metric-lbl'>Fingerprint Dimension</div>"
            f"<div class='metric-desc'>Dimensions of the flat deviation vector (Common Phonemes × 17 features).</div>"
            f"</div>",
            unsafe_allow_html=True
        )
