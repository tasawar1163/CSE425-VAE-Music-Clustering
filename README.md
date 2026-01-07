# CSE425-VAE-Music-Clustering
# Unsupervised Clustering of Hybrid-Language Music using VAEs (CSE425)

## Project Overview
This project performs unsupervised clustering of music tracks using Variational Autoencoders (VAEs).  
We extract audio features from WAV files, learn a latent representation using a VAE, and cluster the tracks in latent space using K-Means.  
We compare the VAE-based clustering against a PCA + K-Means baseline using Silhouette Score and Calinski–Harabasz Index.

**Difficulty Completed:**
- ✅ Easy: MFCC → VAE → KMeans → UMAP/t-SNE + PCA baseline + metrics
- ✅ Medium Extension (partial): CNN-VAE on mel-spectrograms (audio-only)

## Dataset
- Audio format: WAV  
- Number of tracks: 44  
- Language: Hybrid / mixed (English + Bangla / unknown labeling)  
- Note: Raw audio files are not included in this repository due to size and copyright concerns.

## Methods (Summary)

### Easy Pipeline
1. Preprocess audio (mono, resample, trim/pad)
2. Extract MFCC features (mean + std)
3. Baseline: PCA + K-Means
4. Train MLP-VAE on MFCC vectors
5. Extract latent vectors and apply K-Means
6. Evaluate using Silhouette Score and Calinski–Harabasz Index
7. Visualize clusters using UMAP or t-SNE

### Medium Extension (Audio-only)
1. Convert audio to mel-spectrograms
2. Train a CNN-VAE (Conv + ConvTranspose)
3. Extract latent embeddings `z_audio_cnn`

## How to Run (Google Colab)
Open the notebook:
- **[EDIT HERE: add notebook name]** e.g. `CSE425_VAE_Clustering.ipynb`

Run cells in order:
1. Install dependencies
2. Upload WAV files to `data/audio/`
3. Create `metadata.csv`
4. Preprocess audio
5. Extract MFCC features
6. Baseline PCA + KMeans
7. Train VAE
8. Cluster latent vectors
9. Visualize latent clusters
10. Generate final comparison table

## Results

### Metrics Saved
- Baseline metrics: `results/baseline_metrics.csv`
- VAE metrics: `results/vae_metrics.csv`
- Final comparison: `results/final_comparison.csv`

### Visualizations
- VAE latent visualization: `results/latent_visualization/`

**Example plot:**  
![VAE UMAP]([results/latent_visualization/[EDIT_IMAGE_NAME]](https://github.com/tasawar1163/CSE425-VAE-Music-Clustering/blob/main/picture.zip).png)

> Replace `[EDIT_IMAGE_NAME].png` with your real image name.


## Notes / Limitations
- Medium difficulty requires audio + lyrics fusion. This repository includes a CNN-VAE audio-only extension, but lyrics-based embeddings were not used due to dataset limitations.
- Raw audio files are excluded; the pipeline is reproducible by providing WAV files and running the Colab notebook.

## References
- Kingma & Welling, *Auto-Encoding Variational Bayes*, 2014.
- McInnes et al., *UMAP*, 2018.

