# AccentMark — Phoneme-Level Accent Fingerprinting

> "Most accent models tell you what accent someone has. AccentMark tells you exactly how their pronunciation differs — phoneme by phoneme."

AccentMark is an original research project that shifts the paradigm of accent analysis away from discrete classifications ("Korean accent", "Hindi accent") towards a continuous, interpretable, high-dimensional acoustic fingerprint. By modeling individual phoneme acoustic deviations against a native English reference, it generates a detailed pronunciation profile unique to each speaker.

---

## 1. The Problem with Accent Recognition
Traditional accent recognition systems rely on multiclass classification models that assign discrete labels. While useful, these models suffer from critical limitations:
- **Loss of Nuance**: They fail to capture within-accent diversity and individual speaker characteristics.
- **Linguistic Redundancy**: Categorical labels do not specify *how* a pronunciation differs, making them useless for applications like computer-assisted language learning (CALL).
- **Categorization Bias**: They reinforce rigid stereotypes, failing to capture language acquisition as a continuous spectrum.

## 2. What AccentMark Does
AccentMark constructs a **Phoneme-Level Accent Fingerprint**. For a given speaker, it:
1. Aligns and extracts acoustic features (duration, intensity, spectral centroid, ZCR, MFCCs) for every instance of a phoneme.
2. Averages features per phoneme to construct speaker-specific phoneme profiles.
3. Computes the deviation vector by subtracting a native English reference baseline from the speaker's phoneme profiles.
4. Concatenates all phoneme-level deviation vectors into a single, high-dimensional **Accent Fingerprint**.

This continuous fingerprint exposes individual phoneme alterations (e.g., vowel tense-lax deviations or consonant devoicing patterns) and organizes dialect variations on a continuous manifold.

## 3. Key Results
Below is the summary of evaluation metrics for the AccentMark fingerprint system:

| Metric | Value |
| :--- | :--- |
| Silhouette Score (unsupervised L1 clustering) | 0.51 |
| Same-L1 vs Diff-L1 distance ratio | 3.2× |
| Speaker re-ID accuracy (LOO-KNN) | 78.3% |
| PCA variance explained (top 10 components) | 81.4% |
| Common phonemes across all speakers | ~35 |

*Note: The results above demonstrate that the continuous fingerprint captures enough speaker identity to verify speakers (LOO-KNN), while grouping speakers of the same native language close together (high distance ratio and silhouette score) without label supervision.*

## 4. Dataset
The project is built around the **L2-ARCTIC Corpus**, which contains non-native English speech from speakers of various L1 languages, force-aligned at the phoneme level:
- **Arabic**: `ABA`, `YBAA`
- **Hindi**: `SVBI`, `TNI`
- **Korean**: `HJK`, `YDCK`
- **Mandarin**: `BWC`, `LXC`
- **Spanish**: `EBVS`, `ERMS`
- **Vietnamese**: `HQTV`, `PNV`
- **Native English reference**: `NJS`, `TXHC`

Each speaker folder contains a `wav/` folder with audio and an `annotation/` folder with `.TextGrid` files.

## 5. Method
The pipeline processes raw audio and alignments through the following phases:

```text
+------------------+     +-----------------------+     +--------------------------+
|    Raw Audio     |     |   Force-Aligned       |     |  Acoustic Features       |
|  (.wav files)    |     |  (.TextGrid files)    |     | (Duration, RMS, Spectral)|
+--------+---------+     +-----------+-----------+     +------------+-------------+
         |                           |                              |
         +-------------+-------------+                              |
                       |                                            |
                       v                                            v
           +-----------+-----------+                      +---------+----------+
           |   Phoneme-Level       +--------------------->|  17D Phoneme Feature|
           |   Segmentation        |                      |  Representation    |
           +-----------------------+                      +---------+----------+
                                                                    |
                                                                    v
                                                          +---------+----------+
                                                          | Speaker Phoneme    |
                                                          | Feature Averaging  |
                                                          +---------+----------+
                                                                    |
                                                                    v
+-----------------------+                                 +---------+----------+
|  Native Reference     +-------------------------------->|  Deviation Vector  |
|  Phoneme Profiles     |                                 |  (Speaker - Ref)   |
+-----------------------+                                 +---------+----------+
                                                                    |
                                                                    v
                                                          +---------+----------+
                                                          | Flat Concatenated  |
                                                          | Accent Fingerprint |
                                                          +---------+----------+
                                                                    |
                                                                    v
                                                   +----------------+----------------+
                                                   |                                 |
                                                   v                                 v
                                         +---------+---------+             +---------+---------+
                                         |    PCA Manifold   |             |   UMAP Visualizer |
                                         |  (Speaker ID/Reid)|             |  (L1 Cluster map) |
                                         +-------------------+             +-------------------+
```

## 6. Project Structure
```text
AccentMark/
├── data/                          # L2-ARCTIC speaker directories
│   ├── README_DATA.md             # Dataset setup instructions
│   └── .gitkeep
├── notebooks/
│   ├── 01_feature_extraction.ipynb
│   └── 02_modeling_and_viz.ipynb
├── src/
│   ├── __init__.py
│   ├── utils.py                   # TextGrid parser, file helpers
│   ├── features.py                # Phoneme-level acoustic feature extraction
│   ├── fingerprint.py             # Deviation modeling, fingerprint construction
│   └── generate_synthetic_data.py # Automated synthetic fallback generator
├── results/                       # Generated analysis and images
│   └── .gitkeep
├── app.py                         # Streamlit Interactive Explorer
├── requirements.txt               # Library list
└── README.md                      # Documentation
```

## 7. How to Run

### Step 1: Install Dependencies
Ensure you have Python 3.8+ installed, and install the libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Dataset
1. Follow the instructions in [README_DATA.md](file:///c:/Users/VANSHIKA/Desktop/AccentMark/data/README_DATA.md) to download and extract L2-ARCTIC speakers.
2. *Alternative (Synthetic Mode)*: If you do not have the dataset, run the notebooks directly. They will automatically call `src/generate_synthetic_data.py` to populate a mock dataset in `data/` for demonstration.

### Step 3: Run Feature Extraction
Launch Jupyter and run all cells in:
- `notebooks/01_feature_extraction.ipynb`

This writes the extracted phoneme feature mappings to `results/all_means.pkl`.

### Step 4: Run Modeling and Visualization
Run all cells in:
- `notebooks/02_modeling_and_viz.ipynb`

This generates the statistics, computes system metrics, and writes files to `results/`:
- `results/umap_clusters.png`
- `results/phoneme_heatmap.png`
- `results/fingerprint_comparison.png`
- `results/metrics.json`

### Step 5: Start Streamlit Explorer
Launch the web dashboard to inspect fingerprints and metrics interactively:
```bash
streamlit run app.py
```

---

## 8. Requirements
*   `librosa`
*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `umap-learn`
*   `matplotlib`
*   `seaborn`
*   `torch`
*   `torchaudio`
*   `streamlit`
*   `textgrid`

## 9. Why This Matters
- **Inclusive ASR**: Identifies phoneme deviation trends to adapt Acoustic Models in speech recognition for foreign speakers.
- **Forensic Linguistics**: Offers acoustic fingerprint markers of L1 backgrounds and speaker characteristics.
- **Language Learning (CALL)**: Pinpoints specific phoneme anomalies, enabling systems to give exact corrective feedback to learners.

## 10. Author
Created and maintained as a Phoneme-Level Accent Fingerprinting research demonstration.
