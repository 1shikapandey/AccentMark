# L2-ARCTIC Dataset Instructions

To run this project with the real dataset, please follow these steps to download and set up the L2-ARCTIC corpus.

## 1. Download the Dataset
Go to the [L2-ARCTIC Corpus page](https://psi.engr.tamu.edu/l2-arctic-corpus/) and download the zip files for the following speakers:

*   **Arabic**: `ABA`, `YBAA`
*   **Hindi**: `SVBI`, `TNI`
*   **Korean**: `HJK`, `YDCK`
*   **Mandarin**: `BWC`, `LXC`
*   **Spanish**: `EBVS`, `ERMS`
*   **Vietnamese**: `HQTV`, `PNV`
*   **Native English**: `NJS`, `TXHC` (Note: These speakers are from the CMU ARCTIC corpus, also links provided on the same page or can be formatted in the same way).

## 2. Extract and Place Folders
Extract the downloaded zip files and place the speaker directories directly inside this `data/` folder.

The resulting directory structure must look like this:

```text
AccentMark/
└── data/
    ├── ABA/
    │   ├── wav/
    │   │   ├── b0001.wav
    │   │   └── ...
    │   └── annotation/
    │       ├── b0001.TextGrid
    │       └── ...
    ├── YBAA/
    │   ├── wav/
    │   └── annotation/
    ├── ... (other speakers)
    └── README_DATA.md
```

## Note on Synthetic Data
If you run the pipeline (`notebooks/01_feature_extraction.ipynb`) without downloading the dataset first, it will automatically detect the empty folders and offer to generate a small synthetic dataset of simulated speaker files and alignment grids so you can test the code immediately.
